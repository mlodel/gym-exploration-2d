import os
import numpy as np
import gym

gym.logger.set_level(40)
from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.utils.env_utils import run_episode, create_env

import cProfile
import pstats

import multiprocessing

import csv
from datetime import datetime

import matplotlib.pyplot as plt


def main():
    # Instantiate the environment
    # env = gym.make("CollisionAvoidance-v0")
    n_envs = 2
    env, _ = create_env(n_envs=n_envs, subproc=(n_envs > 1))
    # Path to Map
    # mapPath = os.path.abspath(os.path.dirname(__file__)) + "/simple_rooms_no_walls.png"
    # mapPath = os.path.abspath(os.path.dirname(__file__)) + "/../../envs/world_maps/002.png"

    # In case you want to save plots, choose the directory
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/../results"
    # env.set_attr('prediction_model', prediction_model['CV']('CV', 0))
    for i in range(n_envs):
        plot_save_dir = save_path + "/figs_env" + str(i) + "/"
        os.makedirs(plot_save_dir, exist_ok=True)
        # env.set_attr('plot_save_dir', plot_save_dir, i)
        env.env_method("set_plot_save_dir", plot_save_dir, indices=i)
        env.env_method("set_n_env", n_envs, i, True, indices=i)
        # if i != 0:
        #     env.env_method('set_plot_env', False, indices=i)

    obs = env.reset()  # Get agents' initial observations

    env.env_method("set_use_expert_action", 1, True, "ig_greedy", False, 0.0, True)
    # env.env_method("set_use_expert_action", 1, False, "ig_greedy", False, 0.0, False)

    # Repeatedly send actions to the environment based on agents' observations
    n_eps = 1
    num_steps = 128
    max_rewards = []
    max_ig_rewards = []
    ig_rewards = [[] for i in range(n_envs)]
    rewards = [[] for i in range(n_envs)]
    eps_reward = []
    coverage_finished = 0
    n_eps_steps = []
    status = []
    env_ids = []
    n_free_cells_eps = []
    eps_ids = np.zeros(n_envs)
    dummy_action = (
        np.zeros((n_envs, 2))
        if Config.ACTION_SPACE_TYPE == Config.continuous
        else np.zeros((n_envs, 1), dtype=np.uint64)
    )

    for j in range(num_steps * n_eps):

        # dummy_action = 1 * np.array([[2.0, 2.0]])

        obs, reward, game_over, info = env.step(dummy_action)

        for i in range(n_envs):

            rewards[i].append(np.squeeze(reward[i]))
            ig_rewards[i].append(np.squeeze(info[i]["ig_reward"]))

            if game_over[i].any():
                # if info[i]["in_collision"]:
                #     break
                if info[i]["finished_coverage"]:
                    coverage_finished += 1
                #     print("All agents finished!")
                #     # break
                n_eps_steps.append(info[i]["step_num"])
                n_free_cells_eps.append(info[i]["n_free_cells"])
                max_rewards.append(np.max(rewards[i]))
                max_ig_rewards.append(np.max(ig_rewards[i]))
                eps_reward.append(np.sum(rewards[i]))
                status.append(
                    0 if info[i]["ran_out_of_time"] or info[i]["in_collision"] else 1
                )
                eps_ids[i] += 1
                env_ids.append([i, eps_ids[i]])
                rewards[i] = []
                ig_rewards[i] = []
            #     rewards = []
            #     print("Avg Episode Reward: " + str(eps_reward))
    env.reset()
    # max_rewards.append(np.max(rewards))
    # eps_reward = np.sum(np.asarray(rewards)) / n_envs / n_eps
    if len(n_eps_steps) > 0:
        print("Avg Episode Reward: " + str(np.mean(eps_reward)))
        print("Max Step Rewards: " + str(np.max(max_rewards)))
        print("Max IG Step Rewards: " + str(np.max(max_ig_rewards)))
        print("N episodes: " + str(len(eps_reward)))
        print("N finished: " + str(coverage_finished))
        print("Avg Steps per Eps: " + str(np.mean(n_eps_steps)))

    output = np.c_[
        np.asarray(eps_reward),
        np.asarray(n_eps_steps),
        np.asarray(n_free_cells_eps),
        np.asarray(status),
        np.asarray(env_ids),
    ]
    np.savetxt(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)) + "/../results" + "/rewards.csv"
        ),
        output,
        delimiter=",",
    )

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats(os.path.dirname(os.path.realpath(__file__)) + '/experiments/results/' + name + '/stats.prof')


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(
        os.path.dirname(os.path.realpath(__file__)) + "/../results/stats.prof"
    )
