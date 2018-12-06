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
import gym
import torch


import sys
sys.path.append('..')
import utils.logger as logger
from datetime import datetime
from utils.scaler import Scaler
from ppo.models import Policy, ValueFunction


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


def sample(self, policy_mean, policy_logvar, state):
    # compute mean action
    mean_action = policy_mean(state)
    # compute action noise
    action_noise = torch.exp(policy_logvar / 2.0) * torch.randn(policy_logvar.size(0))
    action = mean_action + action_noise
    return action.data.numpy()


def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate
    Args:
        env: ai gym environment (object)
        policy_mean: policy mean (ppo.models.Policy)
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


def run_policy(env, policy_mean, policy_logvar, scaler, logger, episodes=5):
    """ Rollout with Policy and Store Trajectories """


def train(env_name, num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult, init_policy_logvar, seed):
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

    # create policy log-variance
    # logvar_speed is used to 'fool' gradient descent into making faster updates
    # to log-variances. heuristic sets logvar_speed based on action dim.
    logvar_speed = (100 * act_dim ) // 48
    log_vars = torch.Tensor(np.ones(act_dim) * init_policy_logvar, requires_grad=True)

    # create policy mean
    policy = Policy(obs_dim, act_dim, hid1_mult)
    # create value_function
    value_function = ValueFunction(obs_dim, hid1_mult)

    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, episodes=5)
    episode = 0
    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, logger, episodes=batch_size)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        policy.update(observes, actions, advantages, logger)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function



def main():
    # parse arguments
    args = parse_arguments()
    # train loop
    train(**vars(args))


if __name__ == "__main__":
    main()

