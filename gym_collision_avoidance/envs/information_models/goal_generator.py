import numpy as np
from typing import Union

from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.information_models.edfMap import edfMap


class GoalGenerator:
    """
    Generates intermediate high-level goals in training
    """

    def __init__(
        self,
        map_size: tuple,
        max_steps: int,
        max_num_goals: int,
        min_steps_between_goals: int,
        goal_radius: int,
        edf_obj: edfMap,
        rng: np.random.Generator = np.random.default_rng(0),
    ):
        # Init parameters
        self.map_size = np.array(map_size)
        self.max_steps = max_steps
        self.max_num_goals = max_num_goals
        self.min_steps_between_goals = min_steps_between_goals
        self.goal_radius = goal_radius
        self.edf_obj = edf_obj
        self.rng = rng

        # Sample goal sequence
        self.num_goals = rng.integers(self.max_num_goals + 1)
        self.max_steps_between_goals = self.max_steps // self.num_goals
        last_step = 0
        self.goal_steps = []
        self.goals = []
        for i in range(self.num_goals):
            # Sample goal timing
            min_steps = 0 if i == 0 else self.min_steps_between_goals
            next_step = rng.integers(min_steps, self.max_steps_between_goals)
            self.goal_steps.append(next_step + last_step)
            last_step = next_step + last_step

            # Sample goal
            goal_ok = False
            while not goal_ok:
                next_goal = rng.random(2) * self.map_size - self.map_size / 2

                # Check if in obstacle
                edf_val = self.edf_obj.get_edf_value_from_pose(next_goal)
                if edf_val < self.goal_radius:
                    continue

                # Check if too close to previous goal
                if any(
                    [
                        np.linalg.norm(goal - next_goal) < 3 * self.goal_radius
                        for goal in self.goals
                    ]
                ):
                    continue

                # if both checks passed, goal is okay
                goal_ok = True

            self.goals.append(next_goal)

        self.current_goal = None

    def next_goal(self, num_steps) -> bool:
        if num_steps in self.goal_steps:
            self.current_goal = self.goals[0]
            self.goals.remove(self.current_goal)
            self.goal_steps.remove(num_steps)

            return True
        else:
            return False

    def get_goal(self) -> Union[tuple, None]:
        return tuple(self.current_goal)

    # def goal_done(self, goal) -> bool:
    #     if goal in self.current_goals:
    #         self.current_goals.remove(goal)
    #         return True
    #     else:
    #         return False
