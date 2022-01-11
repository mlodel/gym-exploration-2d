import os
import numpy as np
import gym

gym.logger.set_level(40)
from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs.config import Config

import cProfile
import pstats

import multiprocessing

import csv
from datetime import datetime

import matplotlib.pyplot as plt


sims = [
        # {"Ntree": 30, "mcts_cp": 0.5},
        # {"Ntree": 30, "mcts_cp": 1.},
        # {"Ntree": 30, "mcts_cp": 2.},
        # {"Ntree": 60, "mcts_cp": 0.5},
        # {"Ntree": 60, "mcts_cp": 1.},
        # {"Ntree": 60, "mcts_cp": 2.},
        {"Ntree": 30, "mcts_cp": 1., "Ncycles": 5},
        {"Ntree": 30, "mcts_cp": 2., "Ncycles": 5},
        {"Ntree": 30, "mcts_cp": 1., "Ncycles": 10},
        {"Ntree": 30, "mcts_cp": 2., "Ncycles": 10},
        {"Ntree": 30, "mcts_cp": 1., "Ncycles": 15},
        {"Ntree": 30, "mcts_cp": 2., "Ncycles": 15},
]


def main(sim):
    Ntree = sim["Ntree"]
    Ncycles = sim["Ncycles"]
    mcts_cp = sim["mcts_cp"]
    Nsims = 10

    name = "Ntree" + str(Ntree) + "__" + "Ncycles" + str(Ncycles) + "__" + "mcts_cp" + str(mcts_cp) + "__" + "Nsims" \
           + str(Nsims) + "__" + "horizon" + str(4) + "__xdt" + str(5) + "_"

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

    dmcts_agents = [0,1,2]

    cum_reward = [0.0]

    for i in dmcts_agents:
        env.agents[i].policy.set_param(ego_agent=env.agents[i], occ_map=env.map,
                                       map_size=(Config.MAP_WIDTH, Config.MAP_HEIGHT), detect_fov=60.0,
                                       map_res=Config.SUBMAP_RESOLUTION, detect_range=5.0,
                                       Ntree=Ntree, Nsims=Nsims, parallelize_sims=False, mcts_cp=mcts_cp, mcts_horizon=4,
                                       parallelize_agents=False, dt=0.1, xdt=5, mcts_gamma=0.95, Ncycles=Ncycles)

    profiler = cProfile.Profile()
    profiler.enable()

    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 300
    for i in range(num_steps):
        actions = {}
        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, game_over, which_agents_done = env.step(actions)

        cum_reward.append( env.agents[0].policy.team_reward + cum_reward[-1] )

        if game_over:
            print("All agents finished!")
            break
    env.reset()

    with open(
        os.path.dirname(os.path.realpath(__file__)) + '/../results/' + name + '/rewards.csv', 'w+', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(cum_reward)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats(os.path.dirname(os.path.realpath(__file__)) + '/../results/' + name + '/stats.prof')

    return cum_reward, name


if __name__ == '__main__':
    mp_context = multiprocessing.get_context("spawn")
    pool = mp_context.Pool(processes=6)
    results = pool.map(main, sims)
    # results = []
    # for i in range(len(sims)):
    #     results.append(main(sims[i]))

    fig = plt.figure()
    plt.rc('font', size=10)
    fig.set_size_inches(10, 5)
    ax = fig.add_subplot(111)
    for rewards,name in results:
        ax.plot(rewards, label=name)
    ax.set_xlabel('timesteps')
    ax.set_ylabel('cum. rewards [bits]')

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.legend()
    fig.tight_layout()

    dateObj = datetime.now()
    timestamp = dateObj.strftime("%Y%m%d_%H%M%S")
    fig.savefig(os.path.dirname(os.path.realpath(__file__)) + '/../results/__compare/' + timestamp + ".png")