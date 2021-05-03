import os
import numpy as np
import gym

gym.logger.set_level(40)
from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs.config import Config

import cProfile
import pstats

import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None


def main():
    '''
    Minimum working example:
    2 agents: 1 running external policy, 1 running GA3C-CADRL
    '''

    # Create single tf session for all experiments
    # import tensorflow as tf
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # tf.Session().__enter__()

    # Instantiate the environment
    env = gym.make("CollisionAvoidance-v0")

    # Path to Map
    mapPath = os.path.abspath(os.path.dirname(__file__)) + "/simple_rooms_no_walls.png"
    # mapPath = os.path.abspath(os.path.dirname(__file__)) + "/../../envs/world_maps/002.png"

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/test3/')

    # env.set_static_map(mapPath)

    # Set agent configuration (start/goal pos, radius, size, policy)
    # agents, obstacle = tc.agent_with_crossing()
    # [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    # env.set_agents(agents)

    obs = env.reset()  # Get agents' initial observations

    for i in range(3):
        env.agents[i].policy.init_maps(env.agents[i], env.map, (Config.MAP_WIDTH, Config.MAP_HEIGHT), detect_fov=60.0,
                                       map_res=Config.SUBMAP_RESOLUTION, detect_range=5.0)

    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 100
    for i in range(num_steps):

        # Query the external agents' policies
        # e.g., actions[0] = external_policy(dict_obs[0])
        actions = {}
        # actions[0] = np.array([0., 0.0])

        # Internal agents (running a pre-learned policy defined in envs/policies)
        # will automatically query their policy during env.step
        # ==> no need to supply actions for internal agents here

        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, game_over, which_agents_done = env.step(actions)

        if game_over:
            print("All agents finished!")
            break
    env.reset()

    return True


if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('stats6.prof')
