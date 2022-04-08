import gc

import numpy as np

from gym_collision_avoidance.envs.information_models.edfMap import edfMap
from gym_collision_avoidance.envs.information_models.targetMap import targetMap

from gym_collision_avoidance.envs.information_models.ig_greedy import ig_greedy
from gym_collision_avoidance.envs.information_models.ig_mcts import ig_mcts
from gym_collision_avoidance.envs.information_models.ig_random import ig_random

from gym_collision_avoidance.envs.config import Config

import time

def current_milli_time():
    return int(round(time.time() * 1000))

class ig_agent():
    def __init__(self, host_agent, expert_policy):

        self.targetMap = None
        self.detect_fov = None
        self.detect_range = None

        self.obsvd_targets = None
        self.global_pose = None

        self.team_obsv_cells = None
        self.team_reward = None

        self.host_agent = host_agent

        self.expert_goal = np.zeros(2)

        self.expert_policy = expert_policy(self)
        self.expert_seed = 1

        self.agent_pos_map = None
        self.agent_pos_idc = None

        self.finished_entropy = False
        self.finished_binary = False
        self.finished = False
        self.rng = None

        # np.random.seed(current_milli_time() - int(1.625e12))

    def init_model(self, occ_map, map_size, map_res, detect_fov, detect_range, rng):

        self.detect_range = detect_range
        self.detect_fov = detect_fov * np.pi / 180

        # Init EDF and Target Map
        edf_map_obj = edfMap(occ_map, map_res/10, map_size)
        self.targetMap = targetMap(edf_map_obj, map_size, map_res,
                                   sensFOV=self.detect_fov, sensRange=self.detect_range, rOcc=Config.IG_SENSE_rOcc,
                                   rEmp=Config.IG_SENSE_rEmp) # rOcc 3.0 1.1 rEmp 0.33 0.9
        gc.collect()
        self.agent_pos_map = np.zeros(self.targetMap.map.shape)
        self.agent_pos_idc = self.targetMap.getCellsFromPose(self.host_agent.pos_global_frame)
        self.agent_pos_map[self.agent_pos_idc[1], self.agent_pos_idc[0]] = 1.0

        # self.expert_seed = expert_seed
        self.rng = rng


    def set_expert_policy(self, expert):
        if expert == 'ig_greedy':
            self.expert_policy = ig_greedy(self)
        elif expert == 'ig_mcts':
            self.expert_policy = ig_mcts(self)
        elif expert == 'ig_random':
            self.expert_policy = ig_random(self)

    def update(self, agents):

        pos_global_frame = self.host_agent.get_agent_data('pos_global_frame')
        heading_global_frame = self.host_agent.get_agent_data('heading_global_frame')

        global_pose = np.append(pos_global_frame, heading_global_frame)

        ig_agents = [i for i in range(len(agents)) if "ig_" in str(type(agents[i].policy)) and i != self.host_agent.id]
        self.update_belief(ig_agents, global_pose, agents)

        self.agent_pos_map[self.agent_pos_idc[1], self.agent_pos_idc[0]] = 0.0
        self.agent_pos_idc = self.targetMap.getCellsFromPose(self.host_agent.pos_global_frame)
        self.agent_pos_map[self.agent_pos_idc[1], self.agent_pos_idc[0]] = 1.0

        self.finished = self.targetMap.finished

    def get_reward(self, agent_pos, agent_heading):
        return self.targetMap.get_reward_from_pose(np.append(agent_pos, agent_heading))

    def update_belief(self, ig_agents, global_pose, agents):
        targets = []
        poses = []
        # Find Targets in Range and FOV (Detector Emulation)
        self.obsvd_targets = self.find_targets_in_obs(agents, global_pose)
        targets.append(self.obsvd_targets)
        poses.append(global_pose)
        # Get observations of other agens
        for j in ig_agents:
            other_agent_targets = agents[j].policy.obsvd_targets if agents[j].policy.obsvd_targets is not None else []
            targets.append(other_agent_targets)
            other_agent_pose = np.append(agents[j].pos_global_frame, agents[j].heading_global_frame)
            poses.append(other_agent_pose)

        # Update Target Map
        self.team_obsv_cells, self.team_reward = self.targetMap.update(poses, targets, frame='global')
        # self.team_reward = self.targetMap.get_reward_from_cells(self.team_obsv_cells)

    def find_targets_in_obs(self, agents, global_pose):

        # Find Targets in Range and FOV (Detector Emulation)
        c, s = np.cos(global_pose[2]), np.sin(global_pose[2])
        R = np.array(((c, s), (-s, c)))
        targets = []
        for agent in agents:
            if agent.policy.str == "StaticPolicy":
                # Static Agent = Target
                r = agent.pos_global_frame - global_pose[0:2]
                r_norm = np.sqrt(r[0] ** 2 + r[1] ** 2)
                in_range = r_norm <= self.detect_range
                if in_range:
                    r_rot = np.dot(R, r)
                    dphi = (np.arctan2(r_rot[1], r_rot[0]))
                    in_fov = abs(dphi) <= self.detect_fov / 2.0
                    if in_fov:
                        targets.append(agent.pos_global_frame)
                    else:
                        what=1

        return targets

"""
    def get_greedy_goal(self, pose, max_dist=4.0, min_dist=1.0, Nsamples=30):
        # Generate candidate goals in polar coordinates + yaw angle
        candidates_polar = np.random.rand(Nsamples,2)
        # Scale radius
        candidates_polar[:,0] = (max_dist - min_dist) * candidates_polar[:,0] + min_dist
        # Scale angle
        candidates_polar[:,1] = 2*np.pi * candidates_polar[:,1] - np.pi
        # Scale heading angle
        # candidates_polar[:,2] = 2*np.pi * candidates_polar[:,2] - np.pi

        # Convert to xy
        candidates = np.zeros(candidates_polar.shape)
        candidates[:,0] = candidates_polar[:,0] * np.cos(candidates_polar[:,1])
        candidates[:,1] = candidates_polar[:,0] * np.sin(candidates_polar[:,1])
        # candidates[:,2] = candidates_polar[:,2]

        best_cand_idx = None
        max_reward = 0

        # if self.greedy_goal is None:
        #     self.greedy_goal = pose

        for i in range(Nsamples):

            goal = np.append( candidates[i,:] + pose, 0.0)

            # Check if Candidate Goal is Obstacle
            edf_next_pose = self.targetMap.edfMapObj.get_edf_value_from_pose(goal)
            if edf_next_pose < self.host_agent.radius + 0.4:
                reward = 0
            # elif not self.targetMap.edfMapObj.checkVisibility(pose, goal):
            #     reward = 0
            else:
                reward = self.targetMap.get_reward_from_pose(goal)

            if reward >= max_reward:
                max_reward = reward
                best_cand_idx = i

        self.greedy_goal = candidates[best_cand_idx, :] + pose

        return self.greedy_goal
"""
