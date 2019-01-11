import gym
import torch
import torch.nn as nn
import ddqn.config.Hyper as Hyper

class MLP(object):
    def __init__(self, obs_dim, act_dim, hidden_sizes, non_linear):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        input_size = self.obs_dim
        if non_linear == "relu":
            self.non_linear = torch.relu
        elif non_linear == "tanh":
            self.non_linear = torch.tanh
        else:
            raise NotImplementedError
        self.hidden_layers = []
        for i, next_size in enumerate(hidden_sizes):
            self.layers.append(nn.Linear(input_size, next_size))
            input_size = next_size
        self.output_layer = nn.Linear(input_size, self.act_dim)

    def forward(self, observation, action):
        obs = torch.Tensor(observation).to(Hyper.device)
        act = utils.convert_to_one_hot(action, self.act_dim)
        act = torch.Tensor(act).to(Hyper.device)
        hidden = obs
        for layer in self.hidden_layers:
            hidden = self.non_linear(layer(hidden))
        output = self.output_layer(hidden) * act
        return output.cpu().data.numpy()


class ValueFunction(object):
    def __init__(self, mlp, greedy, action_dim):
        self.mlp = mlp
        self.greedy = greedy
        self.action_dim = action_dim



if __name__ == "__main__":
