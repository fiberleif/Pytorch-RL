import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
sys.path.append('..')
import utils.logger as logger
from sklearn.utils import shuffle


class ValueFunction(nn.Module):
    """ NN-based approximation of value function """
    def __init__(self, obs_dim, hid1_mult):
        """
        Args:
            obs_dim: number of dimensions in observation vector (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        super(ValueFunction, self).__init__()
        # hyper-parameters
        self.obs_dim = obs_dim
        self.hid1_mult = hid1_mult
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.epochs = 10
        self.lr = None

        # hid1 layer size is 10x obs_dim, hid3 size is 10, and hid2 is geometric mean
        self.hid1_size = self.obs_dim * self.hid1_mult  # default multipler 10 chosen empirically on 'Hopper-v1'
        self.hid3_size = 5  # 5 chosen empirically on 'Hopper-v1'
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))
        logger.info('Value Params -- h1: {}, h2: {}, h3: {}'.format(self.hid1_size, self.hid2_size, self.hid3_size))

        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 1e-2 / np.sqrt(self.hid2_size)  # 1e-3 empirically determined

        # NN components of value function
        self.fc1 = nn.Linear(self.obs_dim, self.hid1_size)
        self.fc2= nn.Linear(self.hid1_size, self.hid2_size)
        self.fc3 = nn.Linear(self.hid2_size, self.hid3_size)
        self.fc4 = nn.Linear(self.hid3_size, 1)

        # set optimizer
        self._set_optimizer()

    def forward(self, state):
        viewed_state = state.view(-1, self.obs_dim)
        # 3 hidden layers with tanh activations
        x = F.tanh(self.fc1(viewed_state))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

    def _set_optimizer(self):
        self.value_function_optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def _exp_var(self, x, y):
        y_hat = self(torch.Tensor(x))
        y_np = y_hat.detach().numpy()
        exp_var = 1 - np.var(y - y_np) / np.var(y)
        return exp_var

    def update(self, x, y):
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        old_exp_var = self._exp_var(x, y) # test
        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                # placeholder
                x_train_tensor = torch.Tensor(x_train[start:end, :])
                y_train_tensor = torch.Tensor(y_train[start:end]).view(-1, 1)
                # train loss
                loss = nn.MSELoss()
                loss_output = loss(self(x_train_tensor), y_train_tensor)
                self.value_function_optimizer.zero_grad()
                loss_output.backward()
                self.value_function_optimizer.step()
        loss = nn.MSELoss()
        loss_np = loss(self(torch.Tensor(x)), torch.Tensor(y).view(-1, 1)).detach().numpy()
        new_exp_var = self._exp_var(x, y)
        print("vfloss:", loss_np)
        print("oldexpvar:", old_exp_var)
        print("newexpvar:", new_exp_var)

