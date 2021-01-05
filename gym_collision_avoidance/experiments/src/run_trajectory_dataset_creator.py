import os
import numpy as np
import pickle
from tqdm import tqdm
import csv
import json
import atexit
from gym_collision_avoidance.envs.config import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy

np.random.seed(1)

Config.EVALUATE_MODE = False
Config.SAVE_EPISODE_PLOTS = False
Config.SHOW_EPISODE_PLOTS = False
Config.ANIMATE_EPISODES = False
Config.PLOT_CIRCLES_ALONG_TRAJ = False

Config.EVALUATE_MODE =  True
Config.TRAIN_SINGLE_AGENT = False

Config.DT = 0.1
start_from_last_configuration = False

results_subdir = 'free_space_dataset2'


policies = {
            'RVO': {
                'policy': RVOPolicy,
                },
            'NonCooperative': {
                'policy': NonCooperativePolicy
            }
            #'GA3C-CADRL-10': {
            #     'policy': GA3CCADRLPolicy,
            #     'checkpt_dir': 'IROS18',
            #     'checkpt_name': 'network_01900000'
            #     },
            }

num_agents_to_test = [2]
num_test_cases = 3000
test_case_args = {}
Config.NUM_TEST_CASES = num_test_cases


def add_traj(agents, trajs, dt, last_time,writer):
    agent_i = 0
    other_agent_i = (agent_i + 1) % 2

    future_plan_horizon_secs = 3.0
    future_plan_horizon_steps = int(future_plan_horizon_secs / dt)

    for i, agent in enumerate(agents):
        trajectory = []
        max_ts = agent.global_state_history.shape[0]
        for t in range(max_ts):
            #t_horizon = min(max_ts, t+future_plan_horizon_steps)
            #future_linear_speeds = agent.global_state_history[t:t_horizon, 9]
            #future_angular_speeds = agent.global_state_history[t:t_horizon, 10] / dt
            #predicted_cmd = np.dstack([future_linear_speeds, future_angular_speeds])

            #future_positions = agent.global_state_history[t:t_horizon, 1:3]
            other_agents_pos = []
            other_agents_vel = []
            for other_agent in agents:
                if agent.id != other_agent.id:
                    if t >= other_agent.global_state_history.shape[0]:
                        other_agents_pos.append((other_agent.global_state_history[-1, 1],other_agent.global_state_history[-1, 2]))
                        other_agents_vel.append((0,0))
                    else:
                        other_agents_pos.append((other_agent.global_state_history[t, 1],other_agent.global_state_history[t, 2]))
                        other_agents_vel.append((other_agent.global_state_history[t, 7],other_agent.global_state_history[t, 8]))

            d = {'time': np.round(last_time + t*0.1,decimals=1),
            'pedestrian_goal_position': (
                agent.goal_global_frame[0],
                agent.goal_global_frame[1],
            ),
                 'coop_coef': agent.cooperation_coef,
                 'other_agents_pos': other_agents_pos,
                 'other_agents_vel': other_agents_vel,
            'pedestrian_state': {
                 'position': (
                     agent.global_state_history[t, 1],
                     agent.global_state_history[t, 2],
                 ),
                 'velocity': (
                     agent.global_state_history[t, 7],
                     agent.global_state_history[t, 8],
                 )
            },
            }

            #writer.writerow(
            #    [agent.id,np.round(last_time + t*0.1,decimals=1),0,agent.global_state_history[t, 1], agent.global_state_history[t, 2],
            #     agent.global_state_history[t, 7:9], agent.global_state_history[t, 8], agent.goal_global_frame[0], agent.goal_global_frame[1],
            #     agent.cooperation_coef])
            trajectory.append(d)
        trajs.append(trajectory)
    last_time = d['time'] +1.0

    return last_time



file_dir_template = os.path.dirname(os.path.realpath(__file__)) + '/../results/{results_subdir}/{num_agents}_agents'
file_dir = file_dir_template.format(num_agents=10, results_subdir=results_subdir)

"""
def exit_handler():
    pkl_dir = file_dir + '/trajs/'
    os.makedirs(pkl_dir, exist_ok=True)
    # fname = pkl_dir+'RVO.pkl'
    fname = pkl_dir + 'RVO.json'
    # Protocol 2 makes it compatible for Python 2 and 3
    json.dump(trajs, open(fname, 'w'))
    # pickle.dump(trajs, open(fname,'wb'), protocol=2)
    print('dumped {}'.format(fname))

    print("Experiment over.")

atexit.register(exit_handler)

"""

def main():
    env, one_env = create_env()
    dt = one_env.dt_nominal

    last_time = 0.0

    plot_save_dir = file_dir + '/figs/'
    os.makedirs(plot_save_dir, exist_ok=True)

    pkl_dir = file_dir + '/trajs/'
    os.makedirs(pkl_dir, exist_ok=True)

    one_env.plot_save_dir = plot_save_dir
    one_env.scenario = ["single_agents_swap", "single_agents_random_swap","single_agents_random_positions", "single_corridor_scenario"]
    #one_env.scenario = ["agent_with_corridor"]
    one_env.ego_policy = "RVOPolicy"
    one_env.number_of_agents = 5
    env.reset()
    trajs = []
    if not os.path.isfile(file_dir + '/trajs/'+ "dataset_with_images.csv"):
        csvfile = open(file_dir + '/trajs/'+ "dataset.csv", 'w')
    # Write header
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    #writer.writerow(["Id","Time s","Time ns", "Position X",
    #                 "Position Y", "Velocity X", "Velocity Y", "Goal X",
    #                 "Goal Y", "coop_coef"])
    test_case = 0
    pbar = tqdm(total=num_test_cases)
    id = 0
    while test_case < num_test_cases:

        times_to_goal, extra_times_to_goal, collision, all_at_goal, any_stuck, agents = run_episode(env, one_env)

        # Change the global state history according with the number of steps required to finish the episode
        #if all_at_goal:
        for agent in agents:
            agent.global_state_history = agent.global_state_history[:agent.step_num]
        last_time = add_traj(agents, trajs, dt,last_time,writer)
        test_case +=1

        pbar.update(1)

        if (test_case % 500 == 0) and (test_case>8):
            fname = pkl_dir+'RVO'+ str(id) + '.pkl'
            #fname = pkl_dir + 'RVO' + str(id) + '.json'
            # Protocol 2 makes it compatible for Python 2 and 3
            #json.dump(trajs, open(fname, 'a'),indent=test_case)
            pickle.dump(trajs, open(fname,'wb'), protocol=2)
            print('dumped {}'.format(fname))
            trajs = []
            id += 1
    pbar.close()

    print("Experiment over.")

if __name__ == '__main__':
    main()