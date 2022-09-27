"""
Collision Avoidance Environment
Author: Michael Everett
MIT Aerospace Controls Lab
adapted for 2D exploration by Max Lodel, TU Delft Autonomous Multi-Robot Lab
"""

import gym
import gym.spaces
import numpy as np
import itertools
import copy
import os
import time
import multiprocessing

from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.visualize import plot_episode, animate_episode
from gym_collision_avoidance.envs.agent import Agent

from gym_collision_avoidance.envs.maps.map_env import EnvMap
from gym_collision_avoidance.envs.scenario import agent_scenarios
from gym_collision_avoidance.envs.scenario import environment_scenarios

from gym_collision_avoidance.envs.vis_ui.render import GymRenderer


class CollisionAvoidanceEnv(gym.Env):
    metadata = {
        # UNUSED !!
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
    }

    def __init__(self):

        self.id = 0

        # Initialize Rewards
        self._initialize_rewards()

        # Simulation Parameters
        self.num_agents = Config.MAX_NUM_AGENTS_IN_ENVIRONMENT
        self.dt_nominal = Config.DT

        # Collision Parameters
        self.collision_dist = Config.COLLISION_DIST
        self.getting_close_range = Config.GETTING_CLOSE_RANGE

        self.plot_episodes = Config.SHOW_EPISODE_PLOTS or Config.SAVE_EPISODE_PLOTS
        self.plt_limits = Config.PLT_LIMITS
        self.plt_fig_size = Config.PLT_FIG_SIZE
        self.test_case_index = 0

        self.animation_period_steps = Config.ANIMATION_PERIOD_STEPS

        self.number_of_agents = 2
        self.scenario = Config.SCENARIOS_FOR_TRAINING

        self.max_heading_change = 4
        self.min_heading_change = -4
        self.min_speed = -4
        self.max_speed = 4

        self.action_space_type = Config.ACTION_SPACE_TYPE

        if self.action_space_type == Config.discrete:
            if len(Config.DISCRETE_SUBGOAL_RADII) == 1:
                self.action_space = gym.spaces.Discrete(Config.DISCRETE_SUBGOAL_ANGLES)
                radius = Config.DISCRETE_SUBGOAL_RADII[0]
                discrete_angles = np.arange(
                    -np.pi, np.pi, 2 * np.pi / Config.DISCRETE_SUBGOAL_ANGLES
                )
                self.discrete_subgoals = np.asarray(
                    [
                        [radius * np.cos(angle), radius * np.sin(angle)]
                        for angle in discrete_angles
                    ]
                )
            # else:
            #     self.action_space = gym.spaces.MultiDiscrete([])

        elif self.action_space_type == Config.continuous:
            self.low_action = np.array([self.min_speed, self.min_heading_change])
            self.high_action = np.array([self.max_speed, self.max_heading_change])
            self.action_space = gym.spaces.Box(
                self.low_action, self.high_action, dtype=np.float32
            )

        # original observation space
        # self.observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)

        # not used...
        # self.observation_space = np.array([gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)
        # for _ in range(self.num_agents)])
        # observation_space = gym.spaces.Box(self.low_state, self.high_state, dtype=np.float32)

        # single agent dict obs
        self.observation = {}
        for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
            self.observation[agent] = {}

        self.observation_space = gym.spaces.Dict({})
        for state in Config.STATES_IN_OBS:
            if Config.STATE_INFO_DICT[state]["size"] == 1:
                shape = (1,)
            elif isinstance(Config.STATE_INFO_DICT[state]["size"], tuple):
                shape = [1] if len(Config.STATE_INFO_DICT[state]["size"]) < 3 else []
                shape.extend(Config.STATE_INFO_DICT[state]["size"])
                shape = tuple(shape)
            else:
                shape = (Config.STATE_INFO_DICT[state]["size"],)

            self.observation_space.spaces[state] = gym.spaces.Box(
                low=Config.STATE_INFO_DICT[state]["bounds"][0] * np.ones(shape),
                high=Config.STATE_INFO_DICT[state]["bounds"][1] * np.ones(shape),
                shape=shape,
                dtype=Config.STATE_INFO_DICT[state]["dtype"],
            )
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros(
                    (Config.STATE_INFO_DICT[state]["size"]),
                    dtype=Config.STATE_INFO_DICT[state]["dtype"],
                )

        self.agents = None
        self.prev_episode_agents = None

        self.map = None

        self.episode_step_number = 0
        self.episode_number = 0
        self.total_number_of_steps = 0

        self.n_collisions = np.zeros([100])
        self.n_timeouts = np.zeros([100])

        self.plot_save_dir = None
        self.plot_policy_name = None

        self.perturbed_obs = None

        self.obstacles = None

        self.prev_scenario_index = 0
        self.scenario_index = 0

        self.avg_step_time = 0
        self.avg_reset_time = 0

        self.n_other_agents = 0

        self.dagger = False
        self.dagger_beta = 1.0
        self.use_expert = True
        self.expert_controller = "ig_greedy"
        self.comp_expert = True

        self.n_obstacles = Config.TEST_N_OBST

        self.n_env = 1
        self.env_id = 0
        self.plot_every_n_episodes = (
            Config.PLOT_EVERY_N_STEPS
            * Config.REPEAT_STEPS
            // (self.n_env * Config.MAX_TIME_RATIO * 200)
        )

        self.testcase = None
        self.testcase_count = 0
        self.testcase_repeat = 1
        self.testcase_seed = 0
        self.testcase_rng = np.random.default_rng(0)
        self.testcase_n_train = 128
        self.testcase_n_test = 128
        # self.testcases_seeds_train = np.arange(self.testcase_n_train)
        # self.testcases_seeds_test = np.arange(self.testcase_n_train, self.testcase_n_train+self.testcase_n_test)

        self.renderer = None
        self.enable_auto_render = True
        self.auto_rendering_mode = "human"

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True

        self.offscreen = False

        self.run = True
        self.plot_env = True

    def step(self, actions, dt=None):
        ###############################
        # This is the main function. An external process will compute an action for every agent
        # then call env.step(actions). The agents take those actions,
        # then we check if any agents have earned a reward (collision/goal/...).
        # Then agents take an observation of the new world state. We compute whether each agent is done
        # (collided/reached goal/ran out of time) and if everyone's done, the episode ends.
        # We return the relevant info back to the process that called env.step(actions).
        #
        # Inputs
        # - actions: list of [delta heading angle, speed] commands (1 per agent in env)
        # Outputs
        # - next_observations: (obs_length x num_agents) np array with each agent's observation
        # - rewards: list with 1 scalar reward per agent in self.agents
        # - game_over: boolean, true if every agent is done
        # - info_dict: metadata (more details) that help in training, for example
        ###############################

        if dt is None:
            dt = self.dt_nominal

        if self.action_space_type == Config.discrete:
            action_idx = int(actions)
            # action_idx = actions if isinstance(actions, int) else actions[0]
            actions_subgoal = self.discrete_subgoals[action_idx]
        else:
            actions_subgoal = 4.0 * actions

        # self.episode_step_number += 1
        # self.total_number_of_steps += 1

        rewards = 0

        new_action = True

        # Supervisor
        if self.comp_expert:
            start_time = time.time()
            mpc_actions = self.get_expert_goal()
            expert_runtime = time.time() - start_time
        else:
            mpc_actions = np.array([0.0, 0.0])
            expert_runtime = 0

        # Warm-start
        if self.use_expert:
            if self.dagger:
                # LINEAR DECAY
                # self.dagger_beta = np.maximum(self.beta - self.n_env / Config.PRE_TRAINING_STEPS, 0)
                if np.random.uniform(0, 1) > self.dagger_beta:
                    selected_action = actions_subgoal
                else:
                    selected_action = mpc_actions
            else:
                selected_action = mpc_actions
        else:
            self.agents[
                0
            ].policy.enable_collision_avoidance = Config.ENABLE_COLLISION_AVOIDANCE
            selected_action = actions_subgoal

        if self.action_space_type == Config.continuous and Config.CLIP_ACTION:
            clipped_selected_action = np.clip(
                selected_action, self.action_space.low, self.action_space.high
            )
            clipped_actions = np.clip(
                actions_subgoal, self.action_space.low, self.action_space.high
            )
            clipped_mpc_actions = np.clip(
                mpc_actions, self.action_space.low, self.action_space.high
            )
        else:
            clipped_selected_action = selected_action
            clipped_actions = actions_subgoal
            clipped_mpc_actions = mpc_actions

        for i in range(Config.REPEAT_STEPS):

            self.episode_step_number += 1
            self.total_number_of_steps += 1

            # Auto render
            if self.enable_auto_render:
                self.render(mode=self.auto_rendering_mode)

            # Take action
            self._take_action(clipped_selected_action, dt, new_action)
            new_action = False

            if Config.IG_ACCUMULATE_REWARDS or i == Config.REPEAT_STEPS - 1:
                # IG Agents update their models
                for i, agent in enumerate(self.agents):
                    if agent.ig_model is not None:
                        agent.ig_model.update(
                            self.agents, self.episode_step_number // Config.REPEAT_STEPS
                        )

                # Collect rewards
                step_rewards = self._compute_rewards()
                rewards += step_rewards
            # a=b
            if (
                ((self.episode_number - 1) % self.plot_every_n_episodes == 0)
                and not Config.TEST_MODE
                and Config.ANIMATE_EPISODES
                and self.episode_number >= 1
                and self.plot_env
                and self.episode_step_number % self.animation_period_steps == 0
            ):
                plot_episode(
                    self.agents,
                    self.obstacles,
                    self.map,
                    self.episode_number,
                    circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ,
                    plot_save_dir=self.plot_save_dir,
                    plot_policy_name=self.plot_policy_name,
                    save_for_animation=True,
                    limits=self.plt_limits,
                    fig_size=self.plt_fig_size,
                    perturbed_obs=self.perturbed_obs,
                    show=Config.SHOW_EPISODE_PLOTS,
                    save=True,
                )

            # Check which agents' games are finished (at goal/collided/out of time)
            which_agents_done, game_over = self._check_which_agents_done()

        which_agents_done_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]

        # Take observation
        self.map.update(
            pose=np.append(
                self.agents[0].pos_global_frame, self.agents[0].heading_global_frame
            )
        )
        next_observations = self._get_obs()

        infos = {
            "which_agents_done": which_agents_done_dict,
            "is_infeasible": self.agents[0].is_infeasible,
            "is_at_goal": self.agents[0].is_at_goal,
            "step_num": self.agents[0].step_num,
            "ran_out_of_time": self.agents[0].ran_out_of_time,
            "in_collision": self.agents[0].in_collision,
            "deadlocked": self.agents[0].is_deadlocked if game_over else False,
            "subgoal_in_wall": self.agents[0].subgoal_in_wall,
            "n_other_agents": sum(
                [0 if agent.policy.str == "Static" else 1 for agent in self.agents]
            )
            - 1,
            "actions": actions,
            "mpc_actions": clipped_mpc_actions,
            "n_episodes": self.episode_number,
            "ig_reward": self.agents[0].ig_model.team_reward
            if hasattr(self.agents[0], "ig_model")
            else 0.0,
            "finished_coverage": self.agents[0].coverage_finished,
            "n_free_cells": self.agents[0].ig_model.targetMap.n_free_cells,
            "scenario_seed": self.testcase_seed,
            "ig_expert_runtime": expert_runtime,
        }

        return next_observations, rewards, game_over, infos

    def reset(self):

        if self.renderer is not None and self.renderer.is_storing_video:
            self.renderer.save_video()
            self.renderer.is_storing_video = False

        if (
            (
                (self.episode_number - 1) % self.plot_every_n_episodes == 0
                or (
                    Config.TEST_MODE
                    # and self.episode_number <= 2*self.testcase_repeat
                )
            )
            and Config.SAVE_EPISODE_PLOTS
            and self.episode_number >= 1
            and self.episode_step_number > 0
        ):
            plot_episode(
                self.agents,
                self.obstacles,
                self.map,
                self.episode_number,
                self.id,
                circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ,
                plot_save_dir=self.plot_save_dir,
                plot_policy_name=self.plot_policy_name,
                limits=self.plt_limits,
                fig_size=self.plt_fig_size,
                show=Config.SHOW_EPISODE_PLOTS,
                save=Config.SAVE_EPISODE_PLOTS,
            )
            if (not Config.TEST_MODE) and self.plot_env and Config.ANIMATE_EPISODES:
                animate_episode(
                    num_agents=len(self.agents),
                    plot_save_dir=self.plot_save_dir,
                    plot_policy_name=self.plot_policy_name,
                    test_case_index=self.episode_number,
                    agents=self.agents,
                )

        self.episode_number += 1
        self.testcase_count += 1

        self.begin_episode = True
        self.episode_step_number = 0

        self._init_scenario()
        _, collision_with_wall, _, _ = self._check_for_collisions()
        init_pos_infeas = collision_with_wall[0]
        while init_pos_infeas:
            self._init_scenario()
            _, collision_with_wall, _, _ = self._check_for_collisions()
            init_pos_infeas = collision_with_wall[0]

        for agent in self.agents:
            if agent.ig_model is not None:
                agent.ig_model.init_model(
                    map_size=Config.MAP_SIZE,  # self.map.map_size,
                    map_res=Config.IG_MAP_RESOLUTION,
                    detect_fov=Config.IG_SENSE_FOV,
                    detect_range=Config.IG_SENSE_RADIUS,
                    rng=self.testcase_rng,
                    rOcc=Config.IG_SENSE_rOcc,
                    rEmp=Config.IG_SENSE_rEmp,
                    env_map=self.map,
                    init_kwargs=Config.IG_GOALS_SETTINGS,
                )
                # agent.ig_model.update_map(edf_map=self.map.edf_map)
                agent.ig_model.set_expert_policy(self.expert_controller)

        for state in Config.STATES_IN_OBS:
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros(
                    (Config.STATE_INFO_DICT[state]["size"]),
                    dtype=Config.STATE_INFO_DICT[state]["dtype"],
                )

        # IG Agents update their models
        for i, agent in enumerate(self.agents):
            if agent.ig_model is not None:
                agent.ig_model.update(self.agents, 0)

        # Rendering
        self.renderer = GymRenderer(
            self.plot_save_dir, self.map.map_size, self.obstacles, self.map_file
        )

        return self._get_obs()

    def render(self, mode="human"):
        self.renderer.draw_frame(self.agents)
        if mode == "human":
            self.renderer.display()
        elif mode == "rgb_array":
            return self.renderer.get_last_frame()
        elif mode == "save_video":
            self.renderer.is_storing_video = True

    def close(self):
        print("--- Closing CollisionAvoidanceEnv! ---")
        return

    def _take_action(self, actions, dt, new_action=True):
        num_actions_per_agent = self.agents[0].dynamics_model.num_actions
        all_actions = np.zeros(
            (len(self.agents), num_actions_per_agent), dtype=np.float32
        )

        dmcts_agents = []
        # Agents set their action (either from external or w/ find_next_action)
        for agent_index, agent in enumerate(self.agents):
            if agent.is_done:
                continue
            if agent.policy.is_external:
                all_actions[agent_index, :] = agent.policy.convert_to_action(
                    actions[agent_index]
                )
            elif agent.policy.is_still_learning:
                all_actions[agent_index, :] = agent.policy.network_output_to_action(
                    agent_index, self.agents, actions, new_action
                )
            elif "ig_mcts" in str(agent.policy):
                dmcts_agents.append(agent_index)
            else:
                dict_obs = self.observation[agent_index]
                all_actions[agent_index, :] = agent.policy.find_next_action(
                    dict_obs, self.agents, agent_index, self.obstacles
                )

        if len(dmcts_agents) > 0:
            dmcts_actions = self._take_action_dmcts(dmcts_agents)
        for agent_index in dmcts_agents:
            all_actions[agent_index, :] = dmcts_actions[agent_index]

        # After all agents have selected actions, run one dynamics update
        for i, agent in enumerate(self.agents):
            agent.take_action(all_actions[i, :], dt)

    def _take_action_dmcts(self, dmcts_agents):

        actions = {}
        new_step = True
        n_cycles = self.agents[dmcts_agents[0]].policy.Ncycles
        for i in range(n_cycles):
            processes = []
            pipe_list = []
            parallel_agents = []
            mp_context = multiprocessing.get_context("spawn")

            for agent_index in dmcts_agents:
                agent = self.agents[agent_index]
                dict_obs = self.observation[agent_index]
                parallelize = (
                    agent.policy.parallize
                    if hasattr(agent.policy, "parallize")
                    else False
                )
                if parallelize:
                    recv_end, send_end = mp_context.Pipe(False)
                    p = mp_context.Process(
                        target=agent.policy.parallel_next_action,
                        args=(
                            dict_obs,
                            self.agents,
                            agent_index,
                            self.obstacles,
                            send_end,
                            new_step,
                        ),
                    )
                    p.daemon = False
                    processes.append(p)
                    pipe_list.append(recv_end)
                    parallel_agents.append(agent_index)
                    p.start()
                else:
                    actions[agent_index] = agent.policy.find_next_action(
                        dict_obs, self.agents, agent_index, self.obstacles, new_step
                    )
            new_step = False

            for i in range(len(processes)):
                recvd = pipe_list[i].recv()
                pipe_list[i].close()
                processes[i].join()
                self.agents[parallel_agents[i]].policy = recvd["policy_obj"]
                actions[parallel_agents[i]] = recvd["action"]

        return actions

    def _init_scenario(self):

        if Config.TEST_MODE:
            testcases_per_env = self.testcase_n_test // self.n_env
            # if self.testcase_repeat == 1:
            #     self.testcase_count = (self.episode_number - 1)
            tc_start, tc_end = (
                self.testcase_n_train + self.env_id * testcases_per_env,
                self.testcase_n_train + (self.env_id + 1) * testcases_per_env,
            )
            seed = np.arange(tc_start, tc_end)[
                (self.testcase_count - 1) % testcases_per_env
            ]
        else:
            np.random.seed((self.env_id + 1234) * self.episode_number)
            seed = np.random.choice(self.testcase_n_train)

        self.testcase_seed = seed
        self.testcase_rng = np.random.default_rng(seed)

        self.obstacles, self.map_file = getattr(
            environment_scenarios, self.scenario[0]["env"]
        )(Config, n_obstacles=self.n_obstacles, rng=self.testcase_rng)

        self.map = EnvMap(
            map_size=Config.MAP_SIZE,
            cell_size=Config.SUBMAP_RESOLUTION,
            submap_lookahead=Config.SUBMAP_LOOKAHEAD,
            obs_size=Config.EGO_MAP_SIZE,
            obstacles_vert=self.obstacles,
            json=self.map_file,
        )

        self.agents = getattr(agent_scenarios, self.scenario[0]["agents"])(
            Config, env_map=self.map, rng=self.testcase_rng
        )

        for agent in self.agents:
            agent.max_heading_change = self.max_heading_change
            agent.max_speed = self.max_speed
            agent.policy.map = self.map

        self.map.update(
            pose=np.append(
                self.agents[0].pos_global_frame, self.agents[0].heading_global_frame
            )
        )

    def _compute_rewards(self):
        ###############################
        # Check for collisions and reaching of the goal here, and also assign
        # the corresponding rewards based on those calculations.
        #
        # Outputs
        #   - rewards: is a scalar if we are only training on a single agent, or
        #               is a list of scalars if we are training on mult agents
        ###############################

        # if nothing noteworthy happened in that timestep, reward = -0.01
        rewards = self.reward_time_step * np.ones(len(self.agents))
        (
            collision_with_agent,
            collision_with_wall,
            entered_norm_zone,
            dist_btwn_nearest_agent,
        ) = self._check_for_collisions()

        for i, agent in enumerate(self.agents):
            if agent.is_at_goal:
                if agent.was_at_goal_already is False:
                    # agents should only receive the goal reward once
                    rewards[
                        i
                    ] = self.reward_at_goal  # - np.linalg.norm(agent.past_actions[0,:])
                    # print("Agent %i: Arrived at goal!" % agent.id)
            else:
                # collision with other agent
                if agent.was_in_collision_already is False:
                    if collision_with_agent[i]:
                        rewards[i] = self.reward_collision_with_agent
                        agent.in_collision = True
                        # print("Agent %i: Collision with another agent!"
                        #       % agent.id)
                    # collision with a static obstacle
                    elif collision_with_wall[i]:
                        rewards[i] = self.reward_collision_with_wall
                        agent.in_collision = True
                        # print("Agent %i: Collision with wall!"
                        # % agent.id)
                    # There was no collision
                    else:
                        # Penalty for getting close to agents
                        if dist_btwn_nearest_agent[i] <= Config.GETTING_CLOSE_RANGE:
                            rewards[i] += -0.1 - dist_btwn_nearest_agent[i] / 2.0
                            # print("Agent %i: Got close to another agent!"
                            #       % agent.id)
                        # Penalty for wiggly behavior
                        if (
                            np.linalg.norm(
                                agent.past_actions[-1, :] - agent.past_actions[0, :]
                            )
                            > self.wiggly_behavior_threshold
                        ):
                            # Slightly penalize wiggly behavior
                            rewards[i] += self.reward_wiggly_behavior
                        # elif entered_norm_zone[i]:
                        #     rewards[i] = self.reward_entered_norm_zone

                elif agent.ran_out_of_time:
                    if i == 0:
                        # print("Agent 0 is out of time.")
                        pass
                    rewards[i] += Config.REWARD_TIMEOUT

                # If action is infeasible
                if agent.is_infeasible:
                    rewards[i] += Config.REWARD_INFEASIBLE

                # if gets close to goal
                rewards[i] += Config.REWARD_DISTANCE_TO_GOAL * (
                    agent.past_dist_to_goal - agent.dist_to_goal
                )

                if agent.ig_model is not None:
                    # team_reward is reward for last update in ig_model
                    ig_reward = agent.ig_model.team_reward
                    # ig_reward = agent.policy.targetMap.get_reward_from_pose(np.append(agent.pos_global_frame,
                    #                                                                   agent.heading_global_frame))
                    rewards[i] += ig_reward

                if agent.step_num > Config.REPEAT_STEPS:
                    distance = np.linalg.norm(
                        agent.pos_global_frame
                        - agent.global_state_history[
                            : agent.step_num - Config.REPEAT_STEPS, 1:2
                        ]
                    )
                    rewards[i] += distance * Config.REWARD_FACTOR_DISTANCE

                # If subgoal position in inside an obstacle
                """ """
                if i == 0 and not Config.TEST_MODE:
                    agent.subgoal_in_wall = self.map.check_collision(agent.policy.goal_)
                    rewards[i] += (
                        Config.REWARD_SUBGOAL_INFEASIBLE
                        if agent.subgoal_in_wall
                        else 0.0
                    )

                # Penalize Deadlock
                if self.episode_step_number > 20 and (
                    agent.speed_global_frame < 0.01
                    and np.abs(agent.angular_speed_global_frame) < 0.01
                ):
                    rewards[i] += Config.REWARD_DEADLOCKED
                    agent.deadlock_count += 1
                    if agent.deadlock_count > 20:
                        agent.is_deadlocked = True

                # Incentivize Moving
                # rewards[i] += 0.01 * agent.speed_global_frame
        if Config.REWARDS_NORMALIZE:
            rewards = np.clip(
                rewards, self.min_possible_reward, self.max_possible_reward
            ) / (self.max_possible_reward - self.min_possible_reward)

        if Config.TRAIN_SINGLE_AGENT:
            rewards = rewards[0]
        return rewards

    def _check_for_collisions(self):
        collision_with_agent = [False for _ in self.agents]
        collision_with_wall = [False for _ in self.agents]
        entered_norm_zone = [False for _ in self.agents]
        dist_btwn_nearest_agent = [np.inf for _ in self.agents]
        agent_shapes = []
        agent_front_zones = []
        agent_inds = list(range(len(self.agents)))
        agent_pairs = list(itertools.combinations(agent_inds, 2))
        for i, j in agent_pairs:
            agent = self.agents[i]
            other_agent = self.agents[j]
            if (
                "StaticPolicy" in str(type(other_agent.policy))
                and not Config.COLLISION_AV_W_STATIC_AGENT
            ):
                continue
            else:
                dist_btwn = np.linalg.norm(
                    agent.pos_global_frame - other_agent.pos_global_frame
                )
                combined_radius = agent.radius + other_agent.radius
                dist_btwn_nearest_agent[i] = min(
                    dist_btwn_nearest_agent[i], dist_btwn - combined_radius
                )
                if dist_btwn <= combined_radius:
                    # Collision with another agent!
                    collision_with_agent[i] = True
                    collision_with_agent[j] = True
                    if i == 0 and collision_with_agent[i]:
                        # print("Ego-agent collided")
                        pass
        if self.obstacles:
            for i in agent_inds:
                agent = self.agents[i]
                collision_with_wall[i] = self.map.check_collision(
                    agent.pos_global_frame, agent.radius
                )
        else:
            for i in agent_inds:
                collision_with_wall[i] = False

        return (
            collision_with_agent,
            collision_with_wall,
            entered_norm_zone,
            dist_btwn_nearest_agent,
        )

    def _check_which_agents_done(self):
        at_goal_condition = np.array([a.is_at_goal for a in self.agents])
        ran_out_of_time_condition = np.array([a.ran_out_of_time for a in self.agents])
        in_collision_condition = np.array([a.in_collision for a in self.agents])

        if self.agents[0].ig_model is not None:
            if hasattr(self.agents[0].ig_model, "finished"):
                self.agents[0].coverage_finished = self.agents[0].ig_model.finished
                if self.agents[0].coverage_finished:
                    # print("Coverage Finished")
                    pass
                elif self.agents[0].ran_out_of_time:
                    # print("Timeout")
                    pass

        finished_coverage = np.array([a.coverage_finished for a in self.agents])

        which_agents_done = np.logical_or.reduce(
            (
                at_goal_condition,
                ran_out_of_time_condition,
                in_collision_condition,
                finished_coverage,
            )
        )
        for agent_index, agent in enumerate(self.agents):
            agent.is_done = which_agents_done[agent_index]

        if Config.TRAIN_SINGLE_AGENT:
            # Episode ends when ego agent is done
            game_over = which_agents_done[0]
        else:
            # Episode is done when all *learning* agents are done
            learning_agent_inds = [
                i
                for i in range(len(self.agents))
                if self.agents[i].policy.is_still_learning
            ]
            game_over = np.all(which_agents_done[learning_agent_inds])

        return which_agents_done, bool(game_over)

    def _get_obs(self):

        # Agents collect a reading from their map-based sensors
        for i, agent in enumerate(self.agents):
            agent.sense(self.agents, i, self.map)

        # Agents fill in their element of the multiagent observation vector
        for i, agent in enumerate(self.agents):
            self.observation[i] = agent.get_observation_dict(self.agents)

        if Config.TRAIN_SINGLE_AGENT:
            return self.observation[0]
        else:
            return self.observation

    def _initialize_rewards(self):
        self.reward_at_goal = Config.REWARD_AT_GOAL
        self.reward_collision_with_agent = Config.REWARD_COLLISION_WITH_AGENT
        self.reward_collision_with_wall = Config.REWARD_COLLISION_WITH_WALL
        self.reward_getting_close = Config.REWARD_GETTING_CLOSE
        self.reward_time_step = Config.REWARD_TIME_STEP
        self.reward_timeout = Config.REWARD_TIMEOUT
        self.reward_deadlocked = Config.REWARD_DEADLOCKED
        self.reward_subgoal_infeas = Config.REWARD_SUBGOAL_INFEASIBLE

        self.reward_wiggly_behavior = Config.REWARD_WIGGLY_BEHAVIOR
        self.wiggly_behavior_threshold = Config.WIGGLY_BEHAVIOR_THRESHOLD
        self.reward_max_ig = Config.REWARD_MAX_IG
        self.reward_min_ig = Config.REWARD_MIN_IG

        vmax = 2.0
        self.reward_distance_max = (
            vmax * Config.DT * Config.REPEAT_STEPS * Config.REWARD_FACTOR_DISTANCE
        )

        self.possible_terminal_reward_values = np.array(
            [
                self.reward_at_goal,
                self.reward_collision_with_agent,
                self.reward_collision_with_wall,
                self.reward_timeout,
            ]
        )

        self.possible_step_reward_values = [
            self.reward_max_ig,
            self.reward_min_ig,
            self.reward_deadlocked,
            self.reward_subgoal_infeas,
            self.reward_distance_max,
        ]

        repeat_steps = Config.REPEAT_STEPS if Config.IG_ACCUMULATE_REWARDS else 1
        self.min_possible_reward = (
            repeat_steps * self.reward_time_step
            + sum([r if r < 0 else 0 for r in self.possible_step_reward_values])
            + np.min(self.possible_terminal_reward_values)
        )

        self.max_possible_reward = (
            repeat_steps * self.reward_time_step
            + sum([r if r > 0 else 0 for r in self.possible_step_reward_values])
            + np.max(self.possible_terminal_reward_values)
        )

    def get_expert_goal(self):
        if (
            Config.TEST_MODE and not Config.USE_MPC_EXPERT_IN_TEST
        ) or Config.ACTION_SPACE_TYPE == Config.discrete:
            goal = self.agents[0].ig_model.expert_policy.get_expert_goal()[
                0:2
            ]  # - self.agents[0].pos_global_frame
        else:
            goal, exitflag = self.agents[0].policy.mpc_output(0, self.agents)
        return goal

    def set_plot_save_dir(self, plot_save_dir):
        os.makedirs(plot_save_dir, exist_ok=True)
        self.plot_save_dir = plot_save_dir

    def set_plot_env(self, plot_env=True):
        self.plot_env = plot_env

    def set_n_env(self, n_env, env_id, is_val_env):
        self.n_env = n_env
        self.env_id = env_id
        if not is_val_env:
            self.plot_every_n_episodes = int(
                np.ceil(
                    Config.PLOT_EVERY_N_STEPS
                    * Config.REPEAT_STEPS
                    / (self.n_env * Config.MAX_TIME_RATIO * 200)
                )
            )
        else:
            self.plot_every_n_episodes = 1

    def set_use_expert_action(
        self, n_algs, use_expert, expert, dagger, dagger_beta, comp_expert
    ):
        self.use_expert = use_expert
        self.dagger = dagger
        self.dagger_beta = dagger_beta
        self.comp_expert = comp_expert if not use_expert else True
        # TODO check
        if Config.TEST_MODE:
            self.testcase_repeat = n_algs
            self.testcase_count = 0
        if use_expert and n_algs > 0:
            self.expert_controller = expert
            self.agents[0].ig_model.set_expert_policy(expert)

    def set_n_obstacles(self, n_obstacles):
        self.n_obstacles = n_obstacles

    def set_new_human_goal(self, coord: tuple, mode: str = "render"):
        if mode == "render" and self.renderer is not None:
            goal_pos = self.renderer.image_coord_2_map_pos(coord=coord)
            if goal_pos is not None:
                self.agents[0].ig_model.new_human_goal(goal_pos)


if __name__ == "__main__":
    print("See example.py for a minimum working example.")
