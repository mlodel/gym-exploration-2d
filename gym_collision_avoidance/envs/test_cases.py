'''

There are a lot of ways to define a test case.
At the end of the day, you just need to provide a list of Agent objects to the environment.

- For simple testing of a particular configuration, consider defining a function like `get_testcase_two_agents`.
- If you want some final position configuration, consider something like `formation`.
- For a large, static test suite, consider creating a pickle file of [start_x, start_y, goal_x, goal_y, radius, pref_speed] tuples and use our code to convert that into a list of Agents, as in `preset_testCases`.

After defining a test case function that returns a list of Agents, you can select that test case fn in the evaluation code (see example.py)

'''

import numpy as np
import random

# np.random.seed(1)
import sys

sys.path.append('/home/bdebrito/code/mpc-rl-collision-avoidance')
from gym_collision_avoidance.envs.agent import Agent
# from gym_collision_avoidance.envs.Obstacle import Obstacle

from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy


from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy

from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy

from gym_collision_avoidance.envs.policies.MPCRLStaticObsIGPolicy import MPCRLStaticObsIGPolicy
from gym_collision_avoidance.envs.policies.MPCRLStaticObsPolicy import MPCRLStaticObsPolicy
from gym_collision_avoidance.envs.policies.MPCStaticObsPolicy import MPCStaticObsPolicy


from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.dynamics.FirstOrderDynamics import FirstOrderDynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxTurnRate import UnicycleDynamicsMaxTurnRate
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxAcc import UnicycleDynamicsMaxAcc
from gym_collision_avoidance.envs.dynamics.UnicycleSecondOrderEulerDynamics import UnicycleSecondOrderEulerDynamics
from gym_collision_avoidance.envs.dynamics.ExternalDynamics import ExternalDynamics
from gym_collision_avoidance.envs.sensors.OccupancyGridSensor import OccupancyGridSensor
# from gym_collision_avoidance.envs.sensors.AngularMapSensor import AngularMapSensor
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.utils import end_conditions as ec
# from gym_collision_avoidance.envs.dataset import Dataset

from gym_collision_avoidance.envs.information_models.ig_agent import ig_agent
from gym_collision_avoidance.envs.information_models.ig_greedy import ig_greedy
from gym_collision_avoidance.envs.information_models.ig_mcts import ig_mcts

import os
import pickle
import random

test_case_filename = "{dir}/test_cases/{pref_speed_string}{num_agents}_agents_{num_test_cases}_cases.p"


def IG_single_agent():
    pref_speed = 4.0  # np.random.uniform(1.0, 0.5)
    radius = 0.5  # np.random.uniform(0.5, 0.5)
    agents = []

    # Corridor scenario
    obstacle = []

    # ego agent
    agents.append(Agent(0, 0, 0, 10, radius, pref_speed, 0, MPCStaticObsPolicy, FirstOrderDynamics,
                        [OtherAgentsStatesSensor, LaserScanSensor], 0, ig_model=ig_agent))

    return agents, obstacle

def IG_single_agent_crossing(number_of_agents=1, ego_agent_policy=MPCRLStaticObsIGPolicy,
                             other_agents_policy=NonCooperativePolicy, ego_agent_dynamics=FirstOrderDynamics,
                             other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None,
                             n_steps=0, n_env=1, n_obstacles=1):
    pref_speed = 5.0  # np.random.uniform(1.0, 0.5)
    radius = 0.5  # np.random.uniform(0.5, 0.5)
    agents = []
    n_targets = 3
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


    # Corridor scenario
    obstacle = []
    # obstacle_1 = [(2, 6), (-6, 6), (-6, 4), (2, 4)]
    # obstacle_2 = [(7, 7), (5, 7), (5, -4), (7, -4)]
    # obstacle_3 = [(10, -2), (2, -2), (2, -10), (10, -10)]
    # obstacle_4 = [(-2, -1), (-6, -1), (-6, -7), (-2, -7)]

    obstacle_5 = [(-9.8, 10), (-10, 10), (-10, -10), (-9.8, -10)]
    obstacle_6 = [(10, 10), (9.8, 10), (9.8, -10), (10, -10)]
    obstacle_7 = [(-10, 9.8), (-10, 10), (10, 10), (10, 9.8)]
    obstacle_8 = [(10, -9.8), (-10, -9.8), (-10, -10), (10, -10)]

    pos_lims_map = Config.MAP_HEIGHT / 2
    obstacle_margin = 4*radius
    pos_lims_margin = Config.MAP_HEIGHT / 2 - obstacle_margin


    if not Config.TEST_MODE:
        if n_steps < Config.IG_CURRICULUM_LEARNING_STEPS_2_OBS * Config.REPEAT_STEPS/n_env \
                or not Config.IG_CURRICULUM_LEARNING:
            n_obstacles = 1
        elif n_steps < Config.IG_CURRICULUM_LEARNING_STEPS_3_OBS * Config.REPEAT_STEPS/n_env:
            n_obstacles = 2
        else:
            n_obstacles = 3

    obstacle_np = []
    obstacle_at_wall = False

    while len(obstacle) < n_obstacles:
        obst_width = 1.0 * np.random.randint(6, 15)
        obst_height = 1.0 * np.random.randint(1, 8)
        obst_heading = 0.5 * np.pi * np.random.randint(0, 2)
        # if obstacle_at_wall:
        #     obst_center = (2*pos_lims_margin - obst_width) \
        #                   * np.random.rand(2) - pos_lims_margin + obst_width / 2
        # else:
        obst_center = (2*pos_lims_map - obst_width/2) * np.random.rand(2) - pos_lims_map + obst_width/4
        obst_heading = 0.5 * np.pi * np.random.randint(0, 2)


        obstacle_dummy = np.array([[obst_width/2, obst_height/2], [-obst_width/2, obst_height/2],
                    [-obst_width/2, -obst_height/2], [obst_width/2, -obst_height/2]])
        obstacle_shift = obstacle_dummy + (np.ones(obstacle_dummy.shape) * obst_center)
        R = np.array([[np.cos(obst_heading), -np.sin(obst_heading)], [np.sin(obst_heading), np.cos(obst_heading)]])
        obstacle_rot = np.dot(R, obstacle_shift.transpose()).transpose()
        obstacle_rand = [(p[0], p[1]) for p in list(obstacle_rot)]
        obstacle_rand = [obstacle_rand[(i+3)%4] for i in range(4)] if obst_heading != 0.0 else obstacle_rand

        if any([0.2 < pos_lims_map - np.max(np.abs(obstacle_rot[:,i])) < obstacle_margin + 0.2 for i in range(2)]):
            continue
        elif any([0.2 >= pos_lims_map - np.max(np.abs(obstacle_rot[:,i])) for i in range(2)]):
            if obstacle_at_wall:
                continue
            else:
                obstacle_at_wall = True
                obstacle_okay = True
        else:
            obstacle_okay = True

        obstacle_okay = True
        for obst in obstacle_np:
            obstacle_okay = False

            min1, max1 = np.min(obstacle_rot, axis=0), np.max(obstacle_rot, axis=0)
            min2, max2 = np.min(obst, axis=0), np.max(obst, axis=0)
            if ( (0 < min1[0] - max2[0] < obstacle_margin) or (0 < min2[0] - max1[0] < obstacle_margin) \
                    and ((max2[1] - min1[1] > -obstacle_margin/2) and (max1[1] - min2[1] > -obstacle_margin/2)) ) \
                    or \
                    ((max2[0] - min1[0] > -obstacle_margin/2) and (max1[0] - min2[0] > -obstacle_margin/2)) \
                    and (0 < min1[1] - max2[1] < obstacle_margin) or (0 < min2[1] - max1[1] < obstacle_margin):
                break
            else:
                intersecting_area = max(0, min(max1[0], max2[0])
                                        - max(min1[0], min2[0])) \
                                    * max(0, - max(min1[1], min2[1])
                                          + min(max1[1], max2[1]))
                obstacle_area1 = obst_width*obst_height
                obstacle_area2 = (max2[0] - min2[0]) * (max2[1] - min2[1])
                if intersecting_area/obstacle_area1 > 0.3 or intersecting_area/obstacle_area2 > 0.3:
                    break
                else:
                    obstacle_okay=True

        if obstacle_okay:
            obstacle_np.append(obstacle_rot)
            obstacle.extend([obstacle_rand])

    # obstacle.extend([obstacle_1, obstacle_2, obstacle_4, obstacle_5, obstacle_6, obstacle_7, obstacle_8])
    obstacle.extend([obstacle_5, obstacle_6, obstacle_7, obstacle_8])

    # Get random initial position
    pos_infeasible = True
    while pos_infeasible:
        init_pos = (2*pos_lims_margin) * np.random.rand(2) - pos_lims_margin
        init_heading = 2*np.pi * np.random.rand() - np.pi
        pos_infeasible_list = []
        for k in range(n_obstacles):
            obstacle_limits = [[np.min(obstacle_np[k][:,0]), np.max(obstacle_np[k][:,0])],
                               [np.min(obstacle_np[k][:,1]), np.max(obstacle_np[k][:,1])]]
            pos_infeasible_list.append( all([(obstacle_limits[i][0] - radius - 0.2 < init_pos[i] <
                                   obstacle_limits[i][1] + radius + 0.2)
                                for i in range(2)]) )
        pos_infeasible = any(pos_infeasible_list)


    # ego agent
    agents.append(Agent(init_pos[0], init_pos[1], init_pos[0], init_pos[1]+100.0, radius, pref_speed, init_heading,
                        ego_agent_policy, UnicycleSecondOrderEulerDynamics,
                        [OtherAgentsStatesSensor, OccupancyGridSensor], 0, ig_model=ig_agent, ig_expert=ig_greedy))
    # agents.append(Agent(4, 3, 12, 12 + 100.0, radius, pref_speed, - 1*np.pi,
    #                     ego_agent_policy, UnicycleSecondOrderEulerDynamics,
    #                     [OtherAgentsStatesSensor, OccupancyGridSensor], 0, ig_model=ig_agent, ig_expert=ig_greedy))

    # target agent
    for i in range(n_targets):
        pos_infeasible = True
        while pos_infeasible:
            init_pos = (2*pos_lims_margin) * np.random.rand(2) - pos_lims_margin
            pos_infeasible_list = []
            for k in range(n_obstacles):
                obstacle_limits = [[np.min(obstacle_np[k][:, 0]), np.max(obstacle_np[k][:, 0])],
                                   [np.min(obstacle_np[k][:, 1]), np.max(obstacle_np[k][:, 1])]]
                pos_infeasible_list.append(all([(obstacle_limits[i][0] - radius - 0.2 < init_pos[i] <
                                                 obstacle_limits[i][1] + radius + 0.2)
                                                for i in range(2)]))
            pos_infeasible = any(pos_infeasible_list)
        agents.append(Agent(init_pos[0], init_pos[1], 100, 100, 0.2, pref_speed, 0, StaticPolicy, UnicycleSecondOrderEulerDynamics, [], 1))

    if "MPCRLStaticObsPolicy" == str(agents[0].policy) or "MPCStaticObsPolicy" == str(agents[0].policy) \
            or "MPC_IG_Policy" == str(agents[0].policy) or "MPCRLStaticObsIGPolicy" == str(agents[0].policy) \
            or "MPCRLStaticObsIGPolicy_fasttraining" == str(agents[0].policy):
        agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle

def test_scenario(number_of_agents=1, ego_agent_policy=NonCooperativePolicy,
                             other_agents_policy=NonCooperativePolicy, ego_agent_dynamics=FirstOrderDynamics,
                             other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    pref_speed = 4.0  # np.random.uniform(1.0, 0.5)
    radius = 0.5  # np.random.uniform(0.5, 0.5)
    agents = []

    obstacle = []

    # ego agent
    agents.append(Agent(0, 0, 10, 10, radius, pref_speed, 0, MPCStaticObsPolicy, UnicycleSecondOrderEulerDynamics,
                        [OtherAgentsStatesSensor, LaserScanSensor], 0, ig_model=ig_agent))

    agents.append(Agent(10, 0, 0, 0, 0.2, pref_speed, 0, StaticPolicy, FirstOrderDynamics, [], 1))
    agents.append(Agent(-6, -12, 0, 0, 0.2, pref_speed, 0, StaticPolicy, FirstOrderDynamics, [], 2))
    agents.append(Agent(11, 0, 0, 0, 0.2, pref_speed, 0, StaticPolicy, FirstOrderDynamics, [], 3))
    agents.append(Agent(-5, -12, 0, 0, 0.2, pref_speed, 0, StaticPolicy, FirstOrderDynamics, [], 4))
    agents.append(Agent(12, 0, 0, 0, 0.2, pref_speed, 0, StaticPolicy, FirstOrderDynamics, [], 5))
    agents.append(Agent(-4, -12, 0, 0, 0.2, pref_speed, 0, StaticPolicy, FirstOrderDynamics, [], 6))

    if "MPCRLStaticObsPolicy" == str(agents[0].policy) or "MPCStaticObsPolicy" == str(agents[0].policy):
        agents[0].policy.static_obstacles_manager.obstacle = obstacle

    return agents, obstacle


def IG_multi_agent_crossing(number_of_agents=1, ego_agent_policy=NonCooperativePolicy,
                            other_agents_policy=NonCooperativePolicy, ego_agent_dynamics=FirstOrderDynamics,
                            other_agents_dynamics=UnicycleDynamics, agents_sensors=[], seed=None, obstacle=None):
    pref_speed = 1.0  # np.random.uniform(1.0, 0.5)
    radius = 0.5  # np.random.uniform(0.5, 0.5)
    agents = []
    # if seed:
    #     random.seed(seed)
    #     np.random.seed(seed)

    # Corridor scenario
    obstacle = []
    obstacle_1 = [(10, 10), (2, 10), (2, 2), (10, 2)]
    obstacle_2 = [(-2, 10), (-10, 10), (-10, 2), (-2, 2)]
    obstacle_3 = [(10, -2), (2, -2), (2, -10), (10, -10)]
    obstacle_4 = [(-2, -2), (-10, -2), (-10, -10), (-2, -10)]
    obstacle.extend([obstacle_1, obstacle_2, obstacle_3, obstacle_4])

    # ego agent
    agents.append(Agent(-5, 0, 16, 0, radius, pref_speed, 0, ig_mcts, FirstOrderDynamics,
                        [OtherAgentsStatesSensor, LaserScanSensor], 0))
    agents.append(Agent(0, 0, 16, 0, radius, pref_speed, 0, ig_mcts, FirstOrderDynamics,
                        [OtherAgentsStatesSensor, LaserScanSensor], 1))
    agents.append(Agent(5, 0, 16, 0, radius, pref_speed, 0, ig_mcts, FirstOrderDynamics,
                        [OtherAgentsStatesSensor, LaserScanSensor], 2))
    # agents.append(Agent(10, 0, 16, 0, radius, pref_speed, 0, ig_mcts, FirstOrderDynamics,
    #                     [OtherAgentsStatesSensor, LaserScanSensor], 3))

    # target agents
    agents.append(Agent(6, 12, 0, 0, 0.2, pref_speed, 0, StaticPolicy, FirstOrderDynamics, [], 3))
    agents.append(Agent(-6, -12, 0, 0, 0.2, pref_speed, 0, StaticPolicy, FirstOrderDynamics, [], 4))

    return agents, obstacle


if __name__ == '__main__':
    seed = 1
    carrl = False

    # np.random.seed(seed)
    # speed_bnds = [0.5, 1.5]
    speed_bnds = [1.0, 1.0]
    # radius_bnds = [0.2, 0.8]
    radius_bnds = [0.1, 0.1]
    num_agents = 4
    side_length = 4

    ## CARRL
    if carrl:
        num_agents = 2
        side_length = 2
        speed_bnds = [0.5, 1.5]
        radius_bnds = [0.2, 0.8]

    num_test_cases = 500
    test_cases = []

    for i in range(num_test_cases):
        test_case = tc.generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds)
        test_cases.append(test_case)

    if speed_bnds == [1., 1.]:
        pref_speed_string = 'vpref1.0_r{}-{}/'.format(radius_bnds[0], radius_bnds[1])
    else:
        pref_speed_string = ''

    filename = test_case_filename.format(
        num_agents=num_agents, num_test_cases=num_test_cases, pref_speed_string=pref_speed_string,
        dir=os.path.dirname(os.path.realpath(__file__)))
    if carrl:
        filename = filename[:-2] + '_carrl' + filename[-2:]
    filename = filename[:-2] + '_seed' + str(seed).zfill(3) + filename[-2:]

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    pickle.dump(test_cases, open(filename, "wb"))
