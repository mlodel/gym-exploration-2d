import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics

class ExternalDynamics(Dynamics):
    def __init__(self, agent):
        Dynamics.__init__(self, agent)

    def step(self, action, dt):
        # Agents following a trajjectory from a dataset

        step_number = np.minimum(self.agent.step_num, self.agent.policy.trajectory.pose_vec.shape[0] - 1)

        pref_speed = np.linalg.norm(self.agent.policy.trajectory.vel_vec[step_number])
        desired_heading_ego_frame = np.arctan2(self.agent.policy.trajectory.vel_vec[step_number, 0],
                                               self.agent.policy.trajectory.vel_vec[step_number, 1]) \
                                    - self.agent.heading_global_frame

        action = np.array([self.agent.policy.trajectory.pose_vec[step_number, 0],
                           self.agent.policy.trajectory.pose_vec[step_number, 1],
                           desired_heading_ego_frame])

        self.agent.set_state(px=action[0],py=action[1],heading =action[2])
        return