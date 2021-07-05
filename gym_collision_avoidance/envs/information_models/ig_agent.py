import numpy as np

from gym_collision_avoidance.envs.information_models.edfMap import edfMap
from gym_collision_avoidance.envs.information_models.targetMap import targetMap


class ig_agent():
    def __init__(self, host_agent):

        self.targetMap = None
        self.detect_fov = None
        self.detect_range = None

        self.obsvd_targets = None
        self.global_pose = None

        self.team_obsv_cells = None
        self.team_reward = None

        self.host_agent = host_agent

        self.greedy_goal = None

    def init_model(self, occ_map, map_size, map_res, detect_fov, detect_range):

        self.detect_range = detect_range
        self.detect_fov = detect_fov * np.pi / 180

        # Init EDF and Target Map
        edf_map_obj = edfMap(occ_map, map_res, map_size)
        self.targetMap = targetMap(edf_map_obj, map_size, map_res * 5,
                                   sensFOV=self.detect_fov, sensRange=self.detect_range, rOcc=1.1,
                                   rEmp=0.9)

    def update(self, agents):

        pos_global_frame = self.host_agent.get_agent_data('pos_global_frame')
        heading_global_frame = self.host_agent.get_agent_data('heading_global_frame')

        global_pose = np.append(pos_global_frame, heading_global_frame)

        other_agents_states = self.host_agent.get_sensor_data('other_agents_states')

        ig_agents = [i for i in range(len(agents)) if "ig_" in str(type(agents[i].policy)) and i != self.host_agent.id]
        self.update_belief(ig_agents, global_pose, agents, other_agents_states)

    def get_reward(self, agent_pos, agent_heading):
        return self.targetMap.get_reward_from_pose(np.append(agent_pos, agent_heading))

    def update_belief(self, ig_agents, global_pose, agents, other_agents_states):
        targets = []
        poses = []
        # Find Targets in Range and FOV (Detector Emulation)
        self.obsvd_targets = self.find_targets_in_obs(other_agents_states, global_pose)
        targets.append(self.obsvd_targets)
        poses.append(global_pose)
        # Get observations of other agens
        for j in ig_agents:
            other_agent_targets = agents[j].policy.obsvd_targets if agents[j].policy.obsvd_targets is not None else []
            targets.append(other_agent_targets)
            other_agent_pose = np.append(agents[j].pos_global_frame, agents[j].heading_global_frame)
            poses.append(other_agent_pose)

        # Update Target Map
        self.team_obsv_cells = self.targetMap.update(poses, targets, frame='global')
        self.team_reward = self.targetMap.get_reward_from_cells(self.team_obsv_cells)

    def find_targets_in_obs(self, other_agents_states, global_pose):

        # Find Targets in Range and FOV (Detector Emulation)
        targets = []
        for other_agent in other_agents_states:
            if other_agent[9] == 1.0:
                # Static Agent = Target
                # r = other_agent[0:2] - global_pose[0:2]
                r = other_agent[0:2]
                dphi = np.arctan2(r[1], r[0]) - global_pose[2]
                in_fov = abs(dphi) <= self.detect_fov / 2.0
                r_norm = np.sqrt(r[0] ** 2 + r[1] ** 2)
                in_range = r_norm <= self.detect_range
                if in_fov and in_range:
                    targets.append(other_agent[0:2] + global_pose[0:2])

        return targets

    def get_greedy_goal(self, pose, max_dist=5.0, min_dist=2.0, Nsamples=30):
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

        if self.greedy_goal is None:
            self.greedy_goal = pose

        for i in range(Nsamples):

            goal = np.append( candidates[i,:] + pose, 0.0)


            # Check if Candidate Goal is Obstacle
            edf_next_pose = self.targetMap.edfMapObj.get_edf_value_from_pose(goal)
            if edf_next_pose < self.host_agent.radius + 0.1:
                continue
            reward = self.targetMap.get_reward_from_pose(goal)

            if reward > max_reward:
                max_reward = reward
                best_cand_idx = i

        self.greedy_goal = candidates[best_cand_idx, :] + pose

        return self.greedy_goal