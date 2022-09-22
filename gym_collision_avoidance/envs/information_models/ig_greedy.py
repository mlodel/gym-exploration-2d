import numpy as np
from gym_collision_avoidance.envs.config import Config

from gym_collision_avoidance.envs.information_models.targetMap import targetMap


# from gym_collision_avoidance.envs.information_models.ig_agent import ig_agent


class ig_greedy:
    def __init__(self, ig_model):
        self.ig_model = ig_model

    def get_expert_goal(self, max_dist=3.0, min_dist=0.0, Nsamples=30):
        pose = self.ig_model.host_agent.pos_global_frame

        if Config.ACTION_SPACE_TYPE == Config.discrete:
            radius = Config.DISCRETE_SUBGOAL_RADII[0]
            discrete_angles = np.arange(
                -np.pi, np.pi, 2 * np.pi / Config.DISCRETE_SUBGOAL_ANGLES
            )
            candidates = np.asarray(
                [
                    [radius * np.cos(angle), radius * np.sin(angle)]
                    for angle in discrete_angles
                ]
            )
        else:
            # Generate candidate goals in polar coordinates + yaw angle
            candidates_polar = self.ig_model.rng.random((Nsamples, 2))
            # Scale radius
            candidates_polar[:, 0] = (max_dist - min_dist) * candidates_polar[
                :, 0
            ] + min_dist
            # Scale angle
            candidates_polar[:, 1] = 2 * np.pi * candidates_polar[:, 1] - np.pi
            # Scale heading angle
            # candidates_polar[:,2] = 2*np.pi * candidates_polar[:,2] - np.pi

            # Convert to xy
            candidates = np.zeros(candidates_polar.shape)
            candidates[:, 0] = candidates_polar[:, 0] * np.cos(candidates_polar[:, 1])
            candidates[:, 1] = candidates_polar[:, 0] * np.sin(candidates_polar[:, 1])
            # candidates[:,2] = candidates_polar[:,2]

        best_cand_idx = None
        max_reward = 0
        global_goal = None

        # if self.greedy_goal is None:
        #     self.greedy_goal = pose

        c, s = np.cos(self.ig_model.host_agent.heading_global_frame), np.sin(
            self.ig_model.host_agent.heading_global_frame
        )
        R_plus = np.array(((c, -s), (s, c)))

        for i in range(candidates.shape[0]):

            if Config.SUBGOALS_EGOCENTRIC:
                candidate = np.dot(R_plus, candidates[i, :])
            else:
                candidate = candidates[i, :]

            goal = np.append(candidate + pose, 0.0)

            # Check if Candidate Goal is Obstacle
            edf_next_pose = self.ig_model.targetMap.edfMapObj.get_edf_value_from_pose(
                goal
            )
            if edf_next_pose < 0.01:  # self.ig_model.host_agent.radius:
                reward = -1
            elif not self.ig_model.targetMap.edfMapObj.checkVisibility(
                pose, goal  # , thres=self.ig_model.host_agent.radius
            ):
                reward = -1
            else:
                reward = self.ig_model.targetMap.get_reward_from_pose(
                    goal, force_mi=True
                )

            if reward >= max_reward:
                max_reward = reward
                best_cand_idx = i
                global_goal = goal

        if best_cand_idx is not None:
            greedy_goal = candidates[best_cand_idx, :]
        else:
            greedy_goal = np.zeros(2)
        # greedy_goal = candidates[6, :]
        # self.ig_model.expert_goal = global_goal
        # self.ig_model.targetMap.edfMapObj.get_edf_value_from_pose(pose + greedy_goal)
        # self.ig_model.targetMap.edfMapObj.checkVisibility(pose, pose + greedy_goal)

        if greedy_goal.ndim > 1:
            test = 1

        return greedy_goal
