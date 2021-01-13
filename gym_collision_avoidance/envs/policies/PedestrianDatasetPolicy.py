import numpy as np
from gym_collision_avoidance.envs.policies.Policy import Policy

class PedestrianDatasetPolicy(Policy):
    def __init__(self):
        Policy.__init__(self, str="PedestrianDatasetPolicy")

        self.cooperation_coef = 0
        self.agent_id = 0
        self.trajectory = 0

    def __str__(self):
        return "PedestrianDatasetPolicy"

    def find_next_action(self, obs, agents, i, obstacle):
        return np.zeros((2))
