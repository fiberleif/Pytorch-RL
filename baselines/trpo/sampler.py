import torch
import numpy as np


class Sampler(object):
    def __init__(self, env, policy, vf, timesteps_per_batch):
        self.env = env
        self.policy = policy
        self.vf = vf
        self._mb_size = timesteps_per_batch # mini batch size

    def sample(self):
        # Initialize batch arrays
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []

        obs = self.env.reset()
        for t in range(self._mb_size):
            mb_obs.append(obs)
            action = self.policy.select_action(torch.from_numpy(obs), stochastic=True).data.numpy()
            value = self.vf(torch.from_numpy(obs)).data.numpy()
            mb_actions.append(action)
            mb_values.append(value)
            obs, reward, done = self.env.step(action)
            mb_rewards.append(reward)
            bool2float = lambda done: 1.0 if done else 0.0 # maybe bool_to_float
            mb_dones.append(bool2float(done))

        mb_obs = np.asarray(mb_obs, dtype=obs.dtype)
        mb_actions = np.asarray(mb_actions, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.float32)
        last_value = self.vf(torch.from_numpy(obs)).data.numpy()

        return mb_obs, mb_actions, mb_rewards, mb_values, mb_dones, last_value



