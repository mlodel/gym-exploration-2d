import numpy as np
from typing import Union
import random
import copy
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.information_models.edfMap import edfMap

random.seed(0)


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

        # Permute rng to get different goals for each  sequence
        rng_state = self.rng.__getstate__()
        num = random.randint(0, 31)
        new_state = copy.copy(rng_state)
        new_state["state"]["state"] -= num
        self.rng.__setstate__(new_state)

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

            # Sample goal
            goal_ok = False
            failed_trials = 0
            while not goal_ok:

                if failed_trials > 100:
                    break

                next_goal = rng.random(2) * self.map_size - self.map_size / 2
                # Check if in obstacle
                edf_val = self.edf_obj.get_edf_value_from_pose(next_goal)
                if edf_val < self.goal_radius:
                    failed_trials += 1
                    continue

                # Check if too close to previous goal
                if any(
                    [
                        np.linalg.norm(goal - next_goal) < 3 * self.goal_radius
                        for goal in self.goals
                    ]
                ):
                    failed_trials += 1
                    continue

                # if both checks passed, goal is okay
                goal_ok = True

            if goal_ok:
                self.goals.append(next_goal)
                self.goal_steps.append(next_step + last_step)
                last_step = next_step + last_step
            else:
                self.num_goals -= 1

        self.rng.__setstate__(rng_state)

        self.current_goal = None
        self.finished = not (Config.IG_GOALS_TERMINATION and self.num_goals > 0)

    def next_goal(self, num_steps) -> bool:
        if num_steps in self.goal_steps:
            self.current_goal = self.goals[0]
            self.goals.remove(self.current_goal)
            self.goal_steps.remove(num_steps)
            if Config.IG_GOALS_TERMINATION:
                self.finished = len(self.goals) == 0

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
