import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics


class StaticDynamics(Dynamics):
    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.num_actions = 2

    def step(self, action, dt):
        pass
