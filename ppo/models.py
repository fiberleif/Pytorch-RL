import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('..')
import utils.logger as logger


class Policy(nn.Module):
    """ NN-based approximation of policy mean """
    def __init__(self, obs_dim, act_dim, hid1_mult):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        super(Policy, self).__init__()
        # hyper-parameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid1_mult = hid1_mult

        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        self.hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        self.hid3_size = self.act_dim * 10  # 10 empirically determined
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))
        logger.info('Policy Params -- h1: {}, h2: {}, h3: {}'.format(self.hid1_size, self.hid2_size, self.hid3_size))

        # NN components of policy mean
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
    """ NN-based approximation of value function """
    def __init__(self,  obs_dim, hid1_mult):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        super(ValueFunction, self).__init__()
        # hyper-parameters
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult

        # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
        self.hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 chosen empirically on 'Hopper-v1'
        self.hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))
        logger.info('Value Params -- h1: {}, h2: {}, h3: {}'.format(self.hid1_size, self.hid2_size, self.hid3_size))

        # NN components of value function
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


if __name__ == "__main__":
    # unit test for models
    policy = Policy(20, 10, 10)
    value_function = ValueFunction(20, 10)
    # print([1 for name in policy.parameters()])
    # print(type(policy.parameters()))
    # print([name for name, value in policy.state_dict().items()])
    print([name for name, value in value_function.state_dict().items()])






