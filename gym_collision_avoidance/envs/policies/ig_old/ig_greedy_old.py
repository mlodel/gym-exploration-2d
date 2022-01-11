import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy
from gym_collision_avoidance.envs.information_models.edfMap import edfMap
from gym_collision_avoidance.envs.information_models.targetMap import targetMap
class ig_greedy(Policy):
    def __init__(self):
        Policy.__init__(self, str="ig_greedy")

        self.cooperation_coef = 0
        self.edfMap = None
        self.targetMap = None
        self.detect_fov = None
        self.detect_range = None
        self.DT = None

        self.ego_agent = None

    def init_maps(self, ego_agent, occ_map, map_size, map_res, detect_fov, detect_range, dt=0.1):

        self.ego_agent = ego_agent

        self.detect_range = detect_range
        self.detect_fov = detect_fov
        self.DT = dt

        # Init EDF and Target Map
        self.edfMap = edfMap(occ_map, map_res, map_size)
        self.targetMap = targetMap(self.edfMap, map_size, map_res*5,
                                   sensFOV=self.detect_fov * np.pi / 180, sensRange=self.detect_range, rOcc=1.5, rEmp=0.66)

    def find_next_action(self, obs, agents, i, obstacle):

        global_pose = np.append(obs['pos_global_frame'], obs['heading_global_frame'])

        # Find Targets in Range and FOV (Detector Emulation)
        targets = self.find_targets_in_obs(obs)

        # Update Target Map
        self.targetMap.update(global_pose, targets, frame='global')

        # Get Best one-step greedy action
        action = self.greedy_action(global_pose)
        return action

    def find_targets_in_obs(self, obs):
        global_pose = np.append(obs['pos_global_frame'], obs['heading_global_frame'])

        # Find Targets in Range and FOV (Detector Emulation)
        targets = []
        for other_agent in obs['other_agents_states']:
            if other_agent[9] == 1.0:
                # Static Agent = Target
                # r = other_agent[0:2] - global_pose[0:2]
                r = other_agent[0:2]
                dphi = np.arctan2(r[1], r[0]) - global_pose[2]
                in_fov = abs(dphi) <= self.detect_fov / 2.0
                r_norm = np.sqrt(r[0]**2 + r[1]**2)
                in_range = r_norm <= self.detect_range
                if in_fov and in_range:
                    targets.append(other_agent[0:2]+global_pose[0:2])

        return targets

    def greedy_action(self, pose):
        vel_list = [0.0, 2.0, 4.0]
        dphi_list = [-np.pi, 0, np.pi]

        action_list = [np.array([vel,dphi]) for vel in vel_list for dphi in dphi_list]
        max_mi = -1
        best_action = -1
        for action in action_list:
            next_pose = self.get_next_pose(pose, action)
            if next_pose is not None:
                mi = self.targetMap.get_reward_from_pose(next_pose)
                if mi > max_mi:
                    max_mi = mi
                    best_action = action
        return best_action


    def get_next_pose(self, pose, action):
        # Get velocity vector in world frame
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array(((c, -s), (s, c)))
        vel = np.dot(R, np.array([action[0], 0.0]))
        # First Order Dynamics for Next Pose
        dphi = action[1]
        next_pose = pose + np.append(vel, dphi) * self.DT
        # Check if Next Pose is Obstacle
        edf_next_pose = self.targetMap.edfMapObj.get_edf_value_from_pose(next_pose)
        if edf_next_pose > self.ego_agent.radius + 0.1:
            return next_pose
        else:
            return None