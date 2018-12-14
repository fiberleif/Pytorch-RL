import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    """ NN-based approximation of deterministic policy (actor) """
    def __init__(self, obs_dim, act_dim, action_range, actor_lr=1e-4):
        super(Actor, self).__init__()
        # hyperparameter
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_range = action_range
        self.actor_lr = actor_lr

        # NN components
        self.fc1 = nn.Linear(self.obs_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, self.act_dim)

        # set optimizer
        self._set_optimizer()

    def forward(self, observation):
        obs_view = observation.view(-1, self.obs_dim)
        x = F.relu(self.fc1(obs_view))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # scale action
        action = x * self.action_range
        return action

    def _set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.actor_lr)


class Critic(nn.Module):
    """ NN-based approximation of Q function (critic) """
    def __init__(self, obs_dim, act_dim, weight_decay=1e-2):
        super(Critic, self).__init__()
        # hyperparameter
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.weight_decay = weight_decay

        # NN components
        self.fc1 = nn.Linear(self.obs_dim, 400)
        self.fc2 = nn.Linear(400 + self.act_dim, 300)
        self.fc3 = nn.Linear(300, 1)

        # set optimizer
        self._set_optimizer()

    def forward(self, observation, action):
        obs_view = observation.view(-1, self.obs_dim)
        act_view = action.view(-1, self.act_dim)
        x = F.relu(self.fc1(obs_view))
        x = F.relu(self.fc2(torch.cat([x, act_view], 1)))
        x = self.fc3(x)
        return x

    def _set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), weight_decay=self.weight_decay)


class DDPG(object):
    def __init__(self, actor, critic, target_actor, target_critic, noise_std, gamma, tau):
        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.noise_std = noise_std
        self.gamma = gamma
        self.tau = tau

    def select_action(self, observations, add_noise):
        if add_noise:
            action = self.actor(observations).detach().numpy().flatten()
        else:
            noise = np.random.normal(0, self.noise_std, size=self.actor.act_dim)
            action = (self.actor(observations).detach().numpy().flatten() + noise).clip(-self.actor.action_range,
                                                                                        self.actor.action_range)
        return action

    def _compute_target_q(self):
        # compute target Q
        target_actions = self.target_actor(self.next_obs)
        self.target_Q = (self.reward + (1 - self.done) * self.target_critic(self.next_obs, target_actions) * self.gamma).detach()

    def _update_critic(self):
        # build critic loss
        mse_loss = nn.MSELoss()
        critic_loss = mse_loss(self.critic(self.obsveration, self.action), self.target_Q)

        # update critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

    def _update_actor(self):
        # build actor loss
        action = self.actor(self.obsveration)
        actor_loss = -self.critic(self.obsveration, action).mean()

        # update actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

    def update(self, observation, action, reward, next_obs, done):
        # placeholder
        self.obsveration = torch.Tensor(observation)
        self.action = torch.Tensor(action)
        self.reward = torch.Tensor(reward)
        self.next_obs = torch.Tensor(next_obs)
        self.done = torch.Tensor(done)

        # build graph and update models
        self._compute_target_q()
        self._update_critic()
        self._update_actor()

        # update target models
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, directory):
        # save model
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        # load model
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

