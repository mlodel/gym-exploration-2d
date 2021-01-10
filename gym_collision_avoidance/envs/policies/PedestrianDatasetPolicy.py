import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy

class PedestrianDatasetPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="PedestrianDatasetPolicy")

        self.cooperation_coef = 0
        self.trajectory = 0

    def __str__(self):
        return "PedestrianDatasetPolicy"

    def find_next_action(self, obs, agents, i, obstacle):
        # Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents.
        pref_speed = np.linalg.norm(self.trajectory.vel_vec[agents[i].step_num,:2])
        heading_ego_frame = np.arctan2(self.trajectory.vel_vec[agents[i].step_num,1],self.trajectory.vel_vec[agents[i].step_num,0])

        action = np.array([pref_speed, heading_ego_frame])
        return action
