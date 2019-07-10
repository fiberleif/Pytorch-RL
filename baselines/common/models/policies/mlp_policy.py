import torch.nn as nn
import torch
import numpy as np
import baselines.common.utils.distributions as dist


class MLPPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_sizes=(128, 128), activation='tanh',
                 state_dependent_var=True, initial_log_std=0.0, rms=None):
        super(MLPPolicy, self).__init__()
        self._input_dim = input_dim
        self._action_dim = action_dim
        self._state_dependent_var = state_dependent_var
        self._init_log_std = initial_log_std
        self.rms = rms
        if activation == 'tanh':
            self._activation = torch.tanh
        elif activation == 'relu':
            self._activation = torch.relu
        elif activation == 'sigmoid':
            self._activation = torch.sigmoid

        self.hidden_layers = nn.ModuleList()
        last_dim = self._input_dim
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size

        if state_dependent_var:
            self.action = nn.Linear(last_dim, 2 * self._action_dim)
            self.action.weight.data.mul_(0.1)
            self.action.bias.data.mul_(0.0)
        else:
            self.action_mean = nn.Linear(last_dim, self._action_dim)
            # For param initialization.
            self.action_mean.weight.data.mul_(0.1)
            self.action_mean.bias.data.mul_(0.0)
            self.action_log_std = nn.Parameter(torch.ones((1, self._action_dim)) * self._initial_log_std)

    def forward(self, x):
        # Using x, rather than observation (in short, obs), is to support both shared net and non-shared layers.

        # Just for the x = obs case, normalize observation here.
        if self.rms:
            x = self.rms.normalize(x)

        for hidden_layer in self.hidden_layers:
            x = self._activation(hidden_layer(x))

        if self._state_dependent_var:
            action_mean = self.action(x)[:, :self._action_dim]
            action_log_std = self.action(x)[:, self._action_dim:]
        else:
            action_mean = self.action_mean(x)
            action_log_std = self.action_log_std.expand_as(action_mean)

        action_std = torch.exp(action_log_std)
        action_dist = dist.ReinforcedNormal(action_mean, action_std)
        return action_dist

    def select_action(self, obs, stochastic=True):
        action_dist = self.forward(obs)
        if stochastic:
            return action_dist.sample().cpu().data.numpy()
        else:
            return action_dist.mode().cpu().data.numpy()


# test
if __name__ == "__main__":
    input_dim = 20
    action_dim = 10
    states = np.ones((10, input_dim)).astype(np.float32)
    states = torch.from_numpy(states)
    pi = MLPPolicy(input_dim, action_dim)
    actions = pi.select_action(states).data.numpy()
    print(actions.shape)







