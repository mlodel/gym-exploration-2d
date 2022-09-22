import gym
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


def create_env(n_envs=1, subproc=False, seed=0):
    # Generate Environment
    envs = make_vec_env(
        "CollisionAvoidance-v0",
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv if subproc else DummyVecEnv,
    )

    return envs
