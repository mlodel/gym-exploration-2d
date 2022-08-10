import numpy as np
import warnings

from gym_collision_avoidance.envs.information_models.ig_agent import ig_agent
from gym_collision_avoidance.envs.information_models.goal_generator import GoalGenerator

from gym_collision_avoidance.envs.config import Config


class IG_agent_gym(ig_agent):
    def __init__(self, host_agent, expert_policy=None):
        super().__init__(expert_policy)

        self.host_agent = host_agent

        self.global_pose = np.array([0.0, 0.0])

        self.goal_generator = None
        self.goal_radius = None

    def _init_model(
        self,
        max_steps: int,
        max_num_goals: int = 3,
        min_steps_between_goals: int = 10,
        goal_radius: int = 2,
    ):
        if Config.IG_GOALS_ACTIVE:
            self.goal_radius = goal_radius

            self.goal_generator = GoalGenerator(
                map_size=self.map_size,
                max_steps=max_steps,
                max_num_goals=max_num_goals,
                min_steps_between_goals=min_steps_between_goals,
                goal_radius=goal_radius,
                rng=self.rng,
                edf_obj=self.targetMap.edfMapObj,
            )
        else:
            pass

    def update(self, agents, num_steps):

        pos_global_frame = self.host_agent.get_agent_data("pos_global_frame")
        heading_global_frame = self.host_agent.get_agent_data("heading_global_frame")

        self.global_pose = np.append(pos_global_frame, heading_global_frame)

        # Check for new goal
        if Config.IG_GOALS_ACTIVE and not Config.TEST_MODE:
            if self.goal_generator.next_goal(num_steps):
                new_goal = self.goal_generator.get_goal()
                self.targetMap.update_goal_map(new_goal, self.goal_radius)

        self._update_belief(agents)

        self.update_agent_pos_map()

        self.finished = self.targetMap.finished and self.goal_generator.finished

    def new_human_goal(self, new_goal: np.ndarray) -> None:
        if Config.IG_GOALS_ACTIVE:
            self.targetMap.update_goal_map(new_goal, self.goal_radius)
        else:
            warnings.warn(
                "New human goal given, but IG_GOALS_ACTIVE is False. Ignoring new goal ..."
            )

    def _update_belief(self, agents):
        targets = []
        poses = []
        # Find Targets in Range and FOV (Detector Emulation)
        self.obsvd_targets = self._find_targets_in_obs(agents, self.global_pose)
        targets.append(self.obsvd_targets)
        poses.append(self.global_pose)
        # Get observations of other agens
        ig_agents = [
            i
            for i in range(len(agents))
            if "ig_" in str(type(agents[i].policy)) and i != self.host_agent.id
        ]
        for j in ig_agents:
            other_agent_targets = (
                agents[j].policy.obsvd_targets
                if agents[j].policy.obsvd_targets is not None
                else []
            )
            targets.append(other_agent_targets)
            other_agent_pose = np.append(
                agents[j].pos_global_frame, agents[j].heading_global_frame
            )
            poses.append(other_agent_pose)

        # Update Target Map
        self.team_obsv_cells, self.team_reward = self.targetMap.update(
            poses, targets, frame="global"
        )
        # self.team_reward = self.targetMap.get_reward_from_cells(self.team_obsv_cells)

    def _find_targets_in_obs(self, agents, global_pose):

        # Find Targets in Range and FOV (Detector Emulation)
        c, s = np.cos(global_pose[2]), np.sin(global_pose[2])
        R = np.array(((c, s), (-s, c)))
        targets = []
        for agent in agents:
            if agent.policy.str == "StaticPolicy":
                # Static Agent = Target
                r = agent.pos_global_frame - global_pose[0:2]
                r_norm = np.sqrt(r[0] ** 2 + r[1] ** 2)
                in_range = r_norm <= self.detect_range
                if in_range:
                    r_rot = np.dot(R, r)
                    dphi = np.arctan2(r_rot[1], r_rot[0])
                    in_fov = abs(dphi) <= self.detect_fov / 2.0
                    if in_fov:
                        targets.append(agent.pos_global_frame)
                    else:
                        what = 1

        return targets
