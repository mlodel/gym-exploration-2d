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

np.random.seed(1)
import sys

sys.path.append('/home/bdebrito/code/mpc-rl-collision-avoidance')
from gym_collision_avoidance.envs.agent import Agent
# from gym_collision_avoidance.envs.Obstacle import Obstacle
# from gym_collision_avoidance.envs.utils import DataHandlerLSTM
from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
from gym_collision_avoidance.envs.policies.ig_greedy import ig_greedy
from gym_collision_avoidance.envs.policies.ig_mcts import ig_mcts

# from gym_collision_avoidance.envs.policies.PedestrianDatasetPolicy import PedestrianDatasetPolicy
# from gym_collision_avoidance.envs.policies.DRLLongPolicy import DRLLongPolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.CARRLPolicy import CARRLPolicy
from mpc_rl_collision_avoidance.policies.MPCStaticObsPolicy import MPCStaticObsPolicy
from mpc_rl_collision_avoidance.policies.MultiAgentMPCPolicy import MultiAgentMPCPolicy
from mpc_rl_collision_avoidance.policies.OtherAgentMPCPolicy import OtherAgentMPCPolicy
from mpc_rl_collision_avoidance.policies.SocialMPCPolicy import SocialMPCPolicy
from mpc_rl_collision_avoidance.policies.SimpleNNPolicy import SimpleNNPolicy
from mpc_rl_collision_avoidance.policies.MPCRLPolicy import MPCRLPolicy
from mpc_rl_collision_avoidance.policies.LearningMPCPolicy import LearningMPCPolicy
from mpc_rl_collision_avoidance.policies.SafeGA3CPolicy import SafeGA3CPolicy
# from mpc_rl_collision_avoidance.policies.ROSMPCPolicy import ROSMPCPolicy
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.dynamics.FirstOrderDynamics import FirstOrderDynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxTurnRate import UnicycleDynamicsMaxTurnRate
from gym_collision_avoidance.envs.dynamics.UnicycleDynamicsMaxAcc import UnicycleDynamicsMaxAcc
from gym_collision_avoidance.envs.dynamics.UnicycleSecondOrderEulerDynamics import UnicycleSecondOrderEulerDynamics
from gym_collision_avoidance.envs.dynamics.ExternalDynamics import ExternalDynamics
from gym_collision_avoidance.envs.sensors.OccupancyGridSensor import OccupancyGridSensor
from gym_collision_avoidance.envs.sensors.AngularMapSensor import AngularMapSensor
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.utils import end_conditions as ec
# from gym_collision_avoidance.envs.dataset import Dataset

from gym_collision_avoidance.envs.information_models.ig_agent import ig_agent

import os
import pickle
import random

from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import gen_rand_testcases as tc

test_case_filename = "{dir}/test_cases/{pref_speed_string}{num_agents}_agents_{num_test_cases}_cases.p"

policy_dict = {
    'rvo': RVOPolicy,
    'noncoop': NonCooperativePolicy,
    'carrl': CARRLPolicy,
    'external': ExternalPolicy,
    'GA3C': GA3CCADRLPolicy
}

policy_train_dict = {
    '0': RVOPolicy,
    '2': NonCooperativePolicy,
    '1': GA3CCADRLPolicy
}


def IG_single_agent_crossing(number_of_agents=1, ego_agent_policy=NonCooperativePolicy,
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
    agents.append(Agent(-5, 0, 16, 0, radius, pref_speed, 0, NonCooperativePolicy, FirstOrderDynamics,
                        [OtherAgentsStatesSensor, LaserScanSensor], 0, ig_model=ig_agent))

    # target agents
    agents.append(Agent(6, 12, 0, 0, 0.2, pref_speed, 0, StaticPolicy, FirstOrderDynamics, [], 3))
    agents.append(Agent(-6, -12, 0, 0, 0.2, pref_speed, 0, StaticPolicy, FirstOrderDynamics, [], 4))


    if "MPCRLStaticObsPolicy" == str(agents[0].policy):
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

    np.random.seed(seed)
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
