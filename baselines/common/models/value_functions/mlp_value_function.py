import torch
import torch.nn as nn


class MLPValueFunction(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(128, 128), activation='tanh', rms=None):
        super(MLPValueFunction, self).__init__()
        self._input_dim = input_dim
        if activation == 'tanh':
            self._activation = torch.tanh
        elif activation == 'relu':
            self._activation = torch.relu
        elif activation == 'sigmoid':
            self._activation = torch.sigmoid
        self.rms = rms

        self.hidden_layers = nn.ModuleList()
        last_dim = self._input_dim
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size

        self.value = nn.Linear(last_dim, 1)

    def forward(self, x):
        # Just for the x = obs case, normalize observation here.
        if self.rms:
            x = torch.clamp((x - self.rms.mean) / self.rms.std, min(self.rms.clip_range), max(self.rms.clip_range))

        for hidden_layer in self.hidden_layers:
            x = self._activation(hidden_layer(x))
        value = self.value(x)
        return value







