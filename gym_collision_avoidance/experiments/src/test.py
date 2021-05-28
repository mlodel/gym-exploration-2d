import os
import numpy as np
import gym

gym.logger.set_level(40)
from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs.config import Config

import cProfile
import pstats

import multiprocessing

import tensorflow as tf

import csv
from datetime import datetime

import matplotlib.pyplot as plt


if type(tf.contrib) != type(tf): tf.contrib._warning = None


def main(sim):
    name = 'test'
    # Instantiate the environment
    env = gym.make("CollisionAvoidance-v0")

    # Path to Map
    # mapPath = os.path.abspath(os.path.dirname(__file__)) + "/simple_rooms_no_walls.png"
    # mapPath = os.path.abspath(os.path.dirname(__file__)) + "/../../envs/world_maps/002.png"

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/' + name + '/')

    # env.set_static_map(mapPath)

    # Set agent configuration (start/goal pos, radius, size, policy)
    # agents, obstacle = tc.agent_with_crossing()
    # [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    # env.set_agents(agents)

    obs = env.reset()  # Get agents' initial observations

    profiler = cProfile.Profile()
    profiler.enable()

    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 300
    for i in range(num_steps):
        actions = {}
        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, game_over, which_agents_done = env.step(actions)

        if game_over:
            print("All agents finished!")
            break
    env.reset()


    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(os.path.dirname(os.path.realpath(__file__)) + '/../results/' + name + '/stats.prof')

    return name


if __name__ == '__main__':
    main()