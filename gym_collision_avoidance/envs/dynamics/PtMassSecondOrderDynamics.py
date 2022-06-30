import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics
from gym_collision_avoidance.envs.util import wrap, find_nearest
import math


class PtMassSecondOrderDynamics(Dynamics):
    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.max_vel = 2.0
        self.num_actions = 2

    def step(self, action, dt):
        selected_vels = np.clip(
            self.agent.vel_global_frame + action * dt,
            -self.max_vel,
            self.max_vel,
        )

        self.agent.angular_speed_global_frame = 0.0

        selected_heading = wrap(
            self.agent.angular_speed_global_frame * dt + self.agent.heading_global_frame
        )

        self.agent.vel_global_frame[0] = (
            selected_vels[0] * np.cos(selected_heading)
        ) - (selected_vels[1] * np.sin(selected_heading))

        self.agent.vel_global_frame[1] = (
            selected_vels[0] * np.sin(selected_heading)
        ) + (selected_vels[1] * np.cos(selected_heading))

        self.agent.speed_global_frame = np.linalg.norm(selected_vels)

        # dx = selected_vels[0] * np.cos(selected_heading) * dt
        # dy = selected_vels[1] * np.sin(selected_heading) * dt
        self.agent.pos_global_frame += self.agent.vel_global_frame * dt

        self.agent.delta_heading_global_frame = wrap(
            selected_heading - self.agent.heading_global_frame
        )
        self.agent.heading_global_frame = selected_heading
