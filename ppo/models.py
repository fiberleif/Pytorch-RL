import utils.logger as logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Policy(nn.Module):
    """ NN-based policy approximation """
    def __int__(self, obs_dim, act_dim, hid1_mult, policy_logvar):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
            policy_logvar: natural log of initial policy variance
        """

        # hyper-parameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid1_mult = hid1_mult
        self.policy_logvar = policy_logvar

        # tensor components of policy var
        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * self.hid3_size) // 48
        log_vars = torch.zeros((logvar_speed, self.act_dim), requires_grad=True)
        self.log_vars = torch.sum(log_vars, dim=0) + self.policy_logvar

        # NN components of policy mean
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        self.hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        self.hid3_size = self.act_dim * 10  # 10 empirically determined
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))
        logger.info('Policy Params -- h1: {}, h2: {}, h3: {}, logvar_speed: {}'
              .format(self.hid1_size, self.hid2_size, self.hid3_size, logvar_speed))

        self.fc1 = nn.Linear(self.obs_dim, self.hid1_size)
        self.fc2= nn.Linear(self.hid1_size, self.hid2_size)
        self.fc3 = nn.Linear(self.hid2_size, self.hid3_size)
        self.fc4 = nn.Linear(self.hid3_size, self.act_dim)

    def forward(self, state):
        viewed_state = state.view(-1, self.obs_dim)
        # 3 hidden layers with tanh activations
        x = F.tanh(self.fc1(viewed_state))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


class ValueFunction(nn.Module):
    """ NN-based value function approximation """
    def __init__(self,  obs_dim, hid1_mult):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """

        # hyper-parameters
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult

        # NN components of value function
        # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
        self.hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 chosen empirically on 'Hopper-v1'
        self.hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))

        logger.info('Value Params -- h1: {}, h2: {}, h3: {}'
              .format(self.hid1_size, self.hid2_size, self.hid3_size))

        self.fc1 = nn.Linear(self.obs_dim, self.hid1_size)
        self.fc2= nn.Linear(self.hid1_size, self.hid2_size)
        self.fc3 = nn.Linear(self.hid2_size, self.hid3_size)
        self.fc4 = nn.Linear(self.hid3_size, 1)

    def forward(self, state):
        viewed_state = state.view(-1, self.obs_dim)
        # 3 hidden layers with tanh activations
        x = F.tanh(self.fc1(viewed_state))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x






