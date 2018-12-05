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

import gym
import torch
import random
import numpy as np
# from gym import wrappers
# from policy import Policy
# from value_function import NNValueFunction
# from utils import Logger, Scaler
from datetime import datetime
import os
import argparse


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
    parser.add_argument('-v', '--policy_logvar', type=float, default=-1.0,
                        help='Initial policy log-variance (natural log of variance)')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed for all randomness')
    args = parser.parse_args()
    return args

def init_gym(env_name):
    """
    Initialize gym environment
    Args:
        env_name: environment name, e.g. "Humanoid-v1" (str)
    Returns:
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def set_global_seed(seed, env):
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def train(env_name, num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult, policy_logvar, seed):
    """
    Main training loop
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

    # create env component
    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())

    # set seeds
    set_global_seed(seed, env)

    # configure log-information
    cwd = os.path.join(os.getcwd(), 'log')
    run_name = "{0}-{1}".format(env_name, ) + str(args.seed)
    cwd = os.path.join(cwd, run_name)
    logger.configure(dir=cwd)
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories

    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, hid1_mult)
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar)
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

def sample(self, state):
    # compute mean action
    mean_action = self.forward(state)
    # compute action noise
    action_noise =  torch.exp(self.log_vars / 2.0) * torch.randn(self.act_dim)
    sample_action = mean_action + action_noise
    return sample_action




def main():
    # parse arguments
    args = parse_arguments()
    # train loop
    train(**vars(args))


if __name__ == "__main__":
    main()

