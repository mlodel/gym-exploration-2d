import numpy as np

from gym_collision_avoidance.envs.agent import Agent

from gym_collision_avoidance.envs.sensors.global_map_sensor import GlobalMapSensor
from gym_collision_avoidance.envs.sensors.explore_map_sensor import ExploreMapSensor

from gym_collision_avoidance.envs.information_models.ig_agent_gym import IG_agent_gym

from gym_collision_avoidance.envs.policies.MPCRLStaticObsIGPolicy_Drone import (
    MPCRLStaticObsIGPolicy_Drone,
)

from gym_collision_avoidance.envs.policies.go_mpc_drone_decomp import (
    GoMPCDroneDecomp,
)

from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy

from gym_collision_avoidance.envs.dynamics.PtMassSecondOrderDynamics import (
    PtMassSecondOrderDynamics,
)
from gym_collision_avoidance.envs.dynamics.StaticDynamics import StaticDynamics


def exploration_random(Config, env_map, radius=0.15, rng=None, seed=None):
    n_targets = 3

    if seed is not None and rng is None:
        rng = np.random.default_rng(seed)
    elif seed is None and rng is None:
        rng = np.random.default_rng(1)

    agents = []

    pos_lims_margin = np.array(env_map.map_size) / 2 - 2 * radius

    # Get random initial position
    pos_infeasible = True
    while pos_infeasible:
        init_pos = (2 * pos_lims_margin) * rng.random(2) - pos_lims_margin
        init_heading = 2 * np.pi * rng.random() - np.pi
        pos_infeasible = env_map.check_collision(init_pos, radius=radius)

    # ego agent
    agents.append(
        Agent(
            init_pos[0],
            init_pos[1],
            init_pos[0],
            init_pos[1] + 100.0,
            initial_heading=init_heading * 0.0,
            radius=radius,
            pref_speed=0.5,
            policy=GoMPCDroneDecomp,
            dynamics_model=PtMassSecondOrderDynamics,
            sensors=[
                GlobalMapSensor,
                ExploreMapSensor,
            ],
            id=0,
            ig_model=IG_agent_gym,
        )
    )

    # target agents
    target_radius = 0.2
    for i in range(n_targets):
        pos_infeasible = True
        while pos_infeasible:
            init_pos = (2 * pos_lims_margin) * rng.random(2) - pos_lims_margin
            pos_infeasible = env_map.check_collision(init_pos, radius=target_radius)
        agents.append(
            Agent(
                init_pos[0],
                init_pos[1],
                100,
                100,
                target_radius,
                0.5,
                0,
                policy=StaticPolicy,
                dynamics_model=StaticDynamics,
                sensors=[],
                id=1,
            )
        )

    return agents


def exploration_fixed_init(Config, env_map, radius=0.2, rng=None, seed=None):
    agents = []
    # ego agent
    init_heading = 0.0
    init_pos = np.array([-1.5, -3.5])
    agents.append(
        Agent(
            init_pos[0],
            init_pos[1],
            init_pos[0],
            init_pos[1] + 100.0,
            initial_heading=init_heading,
            radius=radius,
            pref_speed=0.5,
            policy=GoMPCDroneDecomp,
            dynamics_model=PtMassSecondOrderDynamics,
            sensors=[
                GlobalMapSensor,
                ExploreMapSensor,
            ],
            id=0,
            ig_model=IG_agent_gym,
        )
    )

    return agents
