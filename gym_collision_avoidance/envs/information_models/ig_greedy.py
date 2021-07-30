import numpy as np

from gym_collision_avoidance.envs.information_models.targetMap import targetMap
# from gym_collision_avoidance.envs.information_models.ig_agent import ig_agent


class ig_greedy():
    def __init__(self, ig_model):
        self.ig_model = ig_model

    def get_expert_goal(self, max_dist=4.0, min_dist=0.0, Nsamples=30):
        pose = self.ig_model.host_agent.pos_global_frame
        # Generate candidate goals in polar coordinates + yaw angle
        # np.random.seed(10)
        candidates_polar = np.random.rand(Nsamples,2)
        # Scale radius
        candidates_polar[:,0] = (max_dist - min_dist) * candidates_polar[:,0] + min_dist
        # Scale angle
        candidates_polar[:,1] = 2*np.pi * candidates_polar[:,1] - np.pi
        # Scale heading angle
        # candidates_polar[:,2] = 2*np.pi * candidates_polar[:,2] - np.pi

        # Convert to xy
        candidates = np.zeros(candidates_polar.shape)
        candidates[:,0] = candidates_polar[:,0] * np.cos(candidates_polar[:,1])
        candidates[:,1] = candidates_polar[:,0] * np.sin(candidates_polar[:,1])
        # candidates[:,2] = candidates_polar[:,2]

        best_cand_idx = None
        max_reward = 0

        # if self.greedy_goal is None:
        #     self.greedy_goal = pose

        for i in range(Nsamples):

            goal = np.append( candidates[i,:] + pose, 0.0)

            # Check if Candidate Goal is Obstacle
            edf_next_pose = self.ig_model.targetMap.edfMapObj.get_edf_value_from_pose(goal)
            if edf_next_pose < self.ig_model.host_agent.radius + 0.1:
                reward = 0
            elif not self.ig_model.targetMap.edfMapObj.checkVisibility(pose, goal):
                reward = 0
            else:
                reward = self.ig_model.targetMap.get_reward_from_pose(goal)

            if reward >= max_reward:
                max_reward = reward
                best_cand_idx = i

        greedy_goal = candidates[best_cand_idx, :] + pose
        self.ig_model.expert_goal = greedy_goal

        return greedy_goal