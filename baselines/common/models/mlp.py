import torch
import torch.nn as nn


class MLPFunction(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(128, 128), activation='tanh', rms=None):
        super(MLPFunction, self).__init__()
        self._obs_dim = obs_dim
        if activation == 'tanh':
            self._activation = torch.tanh
        elif activation == 'relu':
            self._activation = torch.relu
        elif activation == 'sigmoid':
            self._activation = torch.sigmoid
        self.rms = rms

        self.hidden_layers = nn.ModuleList()
        last_dim = obs_dim
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size

    def forward(self, obs):
        if self.rms:
            x = torch.clamp((obs - self.rms.mean) / self.rms.std, min(self.rms.clip_range), max(self.rms.clip_range))
        else:
            x = obs
        for hidden_layer in self.hidden_layers:
            x = self._activation(hidden_layer(x))
        return x

