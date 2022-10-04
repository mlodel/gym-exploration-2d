import gc

import numpy as np

from gym_collision_avoidance.envs.information_models.edfMap import EdfMap
from gym_collision_avoidance.envs.information_models.targetMap import targetMap

from gym_collision_avoidance.envs.information_models.ig_greedy import ig_greedy
from gym_collision_avoidance.envs.information_models.ig_mcts import ig_mcts
from gym_collision_avoidance.envs.information_models.ig_random import ig_random

import time


def current_milli_time():
    return int(round(time.time() * 1000))


class ig_agent:
    def __init__(self, expert_policy=None):

        self.edf_map = None
        self.targetMap = None
        self.detect_fov = None
        self.detect_range = None

        self.obsvd_targets = None

        self.team_obsv_cells = None
        self.team_reward = None

        self.expert_goal = np.zeros(2)

        if expert_policy is not None:
            self.expert_policy = expert_policy(self)
            self.expert_seed = 1
        else:
            self.expert_policy = None

        self.agent_pos_map = None
        self.agent_pos_idc = None

        self.finished_entropy = False
        self.finished_binary = False
        self.finished = False
        self.rng = None

        self.global_pose = np.zeros(3)
        self.map_size = None

        # np.random.seed(current_milli_time() - int(1.625e12))

    def init_model(
        self,
        map_size,
        map_res,
        detect_fov,
        detect_range,
        rng,
        rOcc,
        rEmp,
        env_map,
        init_kwargs=None,
    ):

        if init_kwargs is None:
            init_kwargs = dict()
        self.detect_range = detect_range
        self.detect_fov = detect_fov * np.pi / 180
        self.map_size = map_size
        # Init EDF and Target Map
        self.edf_map = EdfMap(env_map)
        self.targetMap = targetMap(
            self.edf_map,
            map_size,
            map_res,
            sensFOV=self.detect_fov,
            sensRange=self.detect_range,
            rOcc=rOcc,
            rEmp=rEmp,
            real_map_size=env_map.map_size,
        )  # rOcc 3.0 1.1 rEmp 0.33 0.9
        gc.collect()
        self.agent_pos_map = np.zeros(self.targetMap.map.shape)
        self.agent_pos_idc = self.targetMap.getCellsFromPose(self.global_pose)
        self.agent_pos_map[self.agent_pos_idc[0], self.agent_pos_idc[1]] = 1.0

        # self.expert_seed = expert_seed
        self.rng = rng

        init_kwargs = dict if init_kwargs is None else init_kwargs
        self._init_model(**init_kwargs)

    def _init_model(self, **kwargs):
        raise NotImplementedError

    def update_map(self, occ_map=None, edf_map=None):
        self.targetMap.update_map(occ_map, edf_map)

    def set_expert_policy(self, expert):
        if expert == "ig_greedy":
            self.expert_policy = ig_greedy(self)
        elif expert == "ig_mcts":
            self.expert_policy = ig_mcts(self)
        elif expert == "ig_random":
            self.expert_policy = ig_random(self)

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_reward(self, agent_pos, agent_heading):
        return self.targetMap.get_reward_from_pose(np.append(agent_pos, agent_heading))

    def update_agent_pos_map(self):
        self.agent_pos_map[self.agent_pos_idc[0], self.agent_pos_idc[1]] = 0.0
        self.agent_pos_idc = self.targetMap.getCellsFromPose(self.global_pose)
        self.agent_pos_map[self.agent_pos_idc[0], self.agent_pos_idc[1]] = 1.0


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
