import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics
from gym_collision_avoidance.envs.util import wrap, find_nearest
import math

class UnicycleSecondOrderEulerDynamics(Dynamics):
    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.max_turn_rate = 3.0 # rad/s
        self.max_vel = 3.0
        self.num_actions = 2

    def step(self, action, dt):
        selected_speed = np.clip(np.linalg.norm(self.agent.vel_global_frame)+action[0]*dt, 0, self.max_vel)
        turning_rate = self.agent.angular_speed_global_frame +action[1]*dt

        self.agent.angular_speed_global_frame = np.clip(turning_rate, -self.max_turn_rate, self.max_turn_rate)

        selected_heading = wrap(self.agent.angular_speed_global_frame*dt + self.agent.heading_global_frame)

        dx = selected_speed * np.cos(selected_heading) * dt
        dy = selected_speed * np.sin(selected_heading) * dt
        self.agent.pos_global_frame += np.array([dx, dy])

        self.agent.vel_global_frame[0] = selected_speed * np.cos(selected_heading)
        self.agent.vel_global_frame[1] = selected_speed * np.sin(selected_heading)
        self.agent.speed_global_frame = selected_speed
        self.agent.delta_heading_global_frame = wrap(selected_heading -
                                               self.agent.heading_global_frame)
        self.agent.heading_global_frame = selected_heading