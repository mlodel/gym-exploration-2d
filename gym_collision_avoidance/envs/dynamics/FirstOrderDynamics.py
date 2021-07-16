import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics
from gym_collision_avoidance.envs.util import wrap, find_nearest
import math

class FirstOrderDynamics(Dynamics):
    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.num_actions = 2

    def step(self, action, dt):
        selected_speed = action[0]
        selected_heading = wrap(action[1]* dt + self.agent.heading_global_frame)

        dx = selected_speed * np.cos(selected_heading) * dt
        dy = selected_speed * np.sin(selected_heading) * dt
        self.agent.pos_global_frame += np.array([dx, dy])

        self.agent.vel_global_frame[0] = selected_speed * np.cos(selected_heading)
        self.agent.vel_global_frame[1] = selected_speed * np.sin(selected_heading)
        self.agent.speed_global_frame = selected_speed
        self.agent.delta_heading_global_frame = wrap(selected_heading -
                                               self.agent.heading_global_frame)
        self.agent.heading_global_frame = selected_heading