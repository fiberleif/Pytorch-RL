"""
PPO: Proximal Policy Optimization

Written by Guoqing Liu (v-liguoq@microsoft.com)

PPO can be considered as an efficient and reliable improvement to TRPO algorithm.
PPO optimizes a novel "surrogate" objective function with multiple epochs of stochastic gradient ascent.

See these papers for details:
TRPO:
https://arxiv.org/abs/1502.05477 (Schulman et al., 2015)

PPO:
https://arxiv.org/abs/1707.06347 (Schulman et al., 2017)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also these Github repo which was very helpful to me during this implementation:
https://github.com/pat-coady/trpo/

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""

import os
import random
import argparse
import numpy as np
import scipy.signal
import gym
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('..')
import utils.logger as logger
from datetime import datetime
from utils.scaler import Scaler
from policy import Policy
from value_function import ValueFunction


def parse_arguments():
    """ Parse Arguments from Commandline
    Return:
        args: commandline arguments (object)
    """
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('--env_name', type=str, default="HalfCheetah-v1",
                        help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, default=1000,
                        help='Number of episodes to run')
    parser.add_argument('-g', '--gamma', type=float, default=0.995,
                        help='Discount factor')
    parser.add_argument('-l', '--lam', type=float, default=0.98,
                        help='Lambda for Generalized Advantage Estimation')
    parser.add_argument('-k', '--kl_targ', type=float, default=0.003,
                        help='D_KL target value')
    parser.add_argument('-b', '--batch_size', type=int, default=20,
                        help='Number of episodes per training batch')
    parser.add_argument('-t', '--test_frequency', type=int, default=1,
                        help='Number of training batch before test')
    parser.add_argument('-m', '--hid1_mult', type=int, default=10,
                        help='Size of first hidden layer for value and policy NNs'
                             '(integer multiplier of observation dimension)')
    parser.add_argument('-v', '--init_policy_logvar', type=float, default=-1.0,
                        help='Initial policy log-variance (natural log of variance)')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed for all randomness')
    args = parser.parse_args()
    return args


def set_global_seed(seed):
    """ Set Seeds of All Used Modules (Except Env) """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def configure_log_info(env_name, seed):
    """ Configure Log Information """
    cwd = os.path.join(os.getcwd(), 'log')
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique log file
    run_name = "{0}-{1}-{2}".format(env_name, seed, now)
    cwd = os.path.join(cwd, run_name)
    logger.configure(dir=cwd)


def run_episode(env, policy, scaler):
    """ Run single episode with option to animate
    Args:
        env: ai gym environment (object)
        policy_mean: policy mean (ppo.models.Policy)
        policy logvar: policy logvar (torch.tensor)
        scaler:  state scaler, used to scale/offset each observation dimension
            to a similar range (utils.scaler.Scaler)
    Returns:
        observes: shape = (episode len, obs_dim) (np.array)
        actions: shape = (episode len, act_dim) (np.array)
        rewards: shape = (episode len,) (np.array)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim) (np.array)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(np.asarray(reward))
        rewards.append(reward)
        step += 1e-3  # increment time step feature

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, episodes=5):
    """ Rollout with Policy and Store Trajectories """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    # logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
    #             'Steps': total_steps})
    return trajectories, total_steps


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories
    Args:
        trajectories: as returned by run_policy()
        gamma: discount
    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, value_function):
    """ Adds estimated value to all time steps of all trajectories
    Args:
        trajectories: as returned by run_policy()
        value_function: object with predict() method, takes observations
            and returns predicted state value
    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        observes = torch.Tensor(observes)
        values = np.squeeze(value_function(observes).detach().numpy())
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf
    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)
    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """
    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()
    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew):
    """ Log various batch statistics """
    logger.log({'mean_obs': np.mean(observes),
                'min_obs': np.min(observes),
                'max_obs': np.max(observes),
                'std_obs': np.mean(np.var(observes, axis=0)),
                'mean_act': np.mean(actions),
                'min_act': np.min(actions),
                'max_act': np.max(actions),
                'std_act': np.mean(np.var(actions, axis=0)),
                'mean_adv': np.mean(advantages),
                'min_adv': np.min(advantages),
                'max_adv': np.max(advantages),
                'std_adv': np.var(advantages),
                'mean_discrew': np.mean(disc_sum_rew),
                'min_discrew': np.min(disc_sum_rew),
                'max_discrew': np.max(disc_sum_rew),
                'std_discrew': np.var(disc_sum_rew),
                })


def train(env_name, num_episodes, gamma, lam, kl_targ, batch_size, test_frequency,
          hid1_mult, init_policy_logvar, seed):
    """ Main training loop
    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
    """

    # set seeds
    set_global_seed(seed)
    # configure log
    configure_log_info(env_name, seed)

    # create env
    env = gym.make(env_name)
    env.seed(seed) # set env seed
    obs_dim = env.observation_space.shape[0]
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
    act_dim = env.action_space.shape[0]

    # create scaler
    scaler = Scaler(obs_dim)

    # create policy
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, init_policy_logvar)

    # create value_function
    value_function = ValueFunction(obs_dim, hid1_mult)

    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, episodes=5)

    # train & test models
    num_iteration = num_episodes // test_frequency
    current_episodes = 0
    current_steps = 0
    for iter in range(num_iteration):
        # train models
        for i in range(test_frequency):
            # rollout
            trajectories, steps = run_policy(env, policy, scaler, episodes=batch_size)
            # process data
            current_episodes += len(trajectories)
            current_steps += steps
            add_value(trajectories, value_function)  # add estimated values to episodes
            add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
            add_gae(trajectories, gamma, lam)  # calculate advantage
            # concatenate all episodes into single NumPy arrays
            observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
            # add various stats to training log:
            log_batch_stats(observes, actions, advantages, disc_sum_rew)
            # update policy
            policy.update(observes, actions, advantages)  # update policy
            # update value function
            value_function.update(observes, disc_sum_rew)  # update value function

        # test models
        num_test_episodes = 10
        trajectories, _ = run_policy(env, policy, scaler, episodes=num_test_episodes)
        returns = [np.sum(t["rewards"]) for t in trajectories]
        avg_return = sum(returns) / num_test_episodes
        logger.record_tabular('iteration', iter)
        logger.record_tabular('episodes', current_episodes)
        logger.record_tabular('steps', current_steps)
        logger.record_tabular('avg_return', avg_return)
        logger.dump_tabular()


def main():
    # parse arguments
    args = parse_arguments()
    # train loop
    train(**vars(args))


if __name__ == "__main__":
    main()

