import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics

class ExternalDynamics(Dynamics):
    def __init__(self, agent):
        Dynamics.__init__(self, agent)

        self.num_actions = 5

    def step(self, action, dt):
        # Agents following a trajjectory from a dataset

        # step_number = np.minimum(self.agent.step_num, self.agent.policy.trajectory.pose_vec.shape[0] - 1)
        #
        # pref_speed = np.linalg.norm(self.agent.policy.trajectory.vel_vec[step_number])
        # desired_heading_ego_frame = np.arctan2(self.agent.policy.trajectory.vel_vec[step_number, 0],
        #                                        self.agent.policy.trajectory.vel_vec[step_number, 1]) \
        #                             - self.agent.heading_global_frame
        #
        # action = np.array([self.agent.policy.trajectory.pose_vec[step_number, 0],
        #                    self.agent.policy.trajectory.pose_vec[step_number, 1],
        #                    desired_heading_ego_frame])

        if "StaticPolicy" not in str(type(self.agent.policy)):
            self.agent.set_state(px=action[0],py=action[1], vx=action[3], vy=0.0, heading =action[2], ang_speed=action[4])
        return