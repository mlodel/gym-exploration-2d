import gym
import numpy as np
from gym_collision_avoidance.envs.config import Config
from gym_collision_avoidance.envs.wrappers import FlattenDictWrapper, MultiagentFlattenDictWrapper, MultiagentDummyVecEnv
from gym_collision_avoidance.envs.utils.vec_env.dummy_vec_env import DummyVecEnv
from gym_collision_avoidance.envs.utils.vec_env.subproc_vec_env import SubprocVecEnv


def create_env(n_envs=1,eval_env=True, subproc=False):

    
    def make_env():
        env = gym.make("CollisionAvoidance-v0")

        # The env provides a dict observation by default. Most RL code
        # doesn't handle dict observations, so these wrappers convert to arrays
        if Config.TRAIN_SINGLE_AGENT:
            # only return observations of a single agent
            env = FlattenDictWrapper(env, dict_keys=Config.STATES_IN_OBS)
            env=env
        else:
            # return observation of all agents (as a long array)
            env = MultiagentFlattenDictWrapper(env, dict_keys=Config.STATES_IN_OBS, max_num_agents=Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
        
        return env

    # To be prepared for training on multiple instances of the env at once
    if Config.TRAIN_SINGLE_AGENT:
        if subproc:
            env = SubprocVecEnv([make_env for _ in range(n_envs)])
            unwrapped_envs = None
        else:
            env = DummyVecEnv([make_env for _ in range(n_envs)])
            unwrapped_envs = [e.unwrapped for e in env.envs]
    else:
        env = MultiagentDummyVecEnv([make_env for _ in range(n_envs)])
        unwrapped_envs = [e.unwrapped for e in env.envs]

    # Set env id for each env

    if unwrapped_envs is not None:
        for i, e in enumerate(unwrapped_envs):
            e.id = i
        one_env = unwrapped_envs[0]
    else:
        one_env = None

    return env, one_env

def run_episode(env, one_env):
    score = 0
    done = False
    steps = 0
    while not done:
        obs, rew, done, info = env.step([None])
        score += rew[0]
        steps += 1

    # After end of episode, compute statistics about the agents
    agents = one_env.prev_episode_agents
    # agents = one_env.prev_episode_agents
    time_to_goal = np.array([a.t for a in agents])
    extra_time_to_goal = np.array([a.t - a.straight_line_time_to_reach_goal for a in agents])
    collision = np.array(
        np.any([a.in_collision for a in agents])).tolist()
    all_at_goal = np.array(
        np.all([a.is_at_goal for a in agents])).tolist()
    any_stuck = np.array(
        np.any([not a.in_collision and not a.is_at_goal for a in agents])).tolist()

    return time_to_goal, extra_time_to_goal, collision, all_at_goal, any_stuck, agents

def store_stats(stats, policy, test_case, times_to_goal, extra_times_to_goal, collision, all_at_goal, any_stuck):
    stats[policy][test_case] = {}
    stats[policy][test_case]['times_to_goal'] = times_to_goal
    stats[policy][test_case]['extra_times_to_goal'] = extra_times_to_goal
    stats[policy][test_case]['mean_extra_time_to_goal'] = np.mean(extra_times_to_goal)
    stats[policy][test_case]['total_time_to_goal'] = np.sum(times_to_goal)
    stats[policy][test_case]['collision'] = collision
    stats[policy][test_case]['all_at_goal'] = all_at_goal
    if not collision: stats[policy]['non_collision_inds'].append(test_case)
    if all_at_goal: stats[policy]['all_at_goal_inds'].append(test_case)
    if not collision and not all_at_goal: stats[policy]['stuck_inds'].append(test_case)
    return stats
