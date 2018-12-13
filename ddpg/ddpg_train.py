"""
DDPG: Deep Deterministic Policy Gradient

Written by Guoqing Liu (v-liguoq@microsoft.com)

DDPG is a powerful actor-critic algorithm based on the determinstic policy gradient.

See these papers for details:
DPG:
http://proceedings.mlr.press/v32/silver14.pdf (David Silver et al., 2015)

DDPG:
https://arxiv.org/abs/1707.06347 (Timothy P. Lillicrap et al., 2016)

And, also these Github repo which was very helpful to me during this implementation:
https://github.com/sfujim/TD3

This implementation learns policies for continuous environments in the OpenAI Gym (https://gym.openai.com/).
Testing was focused on the MuJoCo control Suite.
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
from models import Actor, Critic, DDPG


def parse_arguments():
    """ Parse Arguments from Commandline
    Return:
        args: commandline arguments (object)
    """
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('-e', '--env_name', type=str, default="HalfCheetah-v1",
                        help='OpenAI Gym environment name')
    parser.add_argument("--start_episodes", type=int, default=10,
                        help='How many episodes purely random policy is run for')
    parser.add_argument('-n', '--num_episodes', type=int, default=1000,
                        help='Number of episodes to run')
    parser.add_argument('-g', '--gamma', type=float, default=0.995,
                        help='Discount factor')
    parser.add_argument('-t', '--tau', type=float, default=0.005,
                        help='Target network update rate')
    parser.add_argument('-b', '--batch_size', type=int, default=100,
                        help='Number of episodes per training batch')
    parser.add_argument('-f', '--eval_freq', type=int, default=10,
                        help='Number of training batch before test')
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








