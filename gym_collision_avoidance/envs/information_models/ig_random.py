import numpy as np

from gym_collision_avoidance.envs.information_models.targetMap import targetMap
# from gym_collision_avoidance.envs.information_models.ig_agent import ig_agent


class ig_random():
    def __init__(self, ig_model):
        self.ig_model = ig_model

    def get_expert_goal(self):
        pose = self.ig_model.host_agent.pos_global_frame
        goal = np.random.rand(2)
        goal = np.array([8., 8.]) * goal - np.array([4., 4.])

        goal += pose

        return goal