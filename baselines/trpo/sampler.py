import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sampler(object):
    def __init__(self, env, policy, value_net, timesteps_per_batch):
        self.env = env
        self.policy = policy
        self.value_net = value_net
        self._mb_size = timesteps_per_batch  # mini batch size

    def sample(self):
        # Initialize batch arrays
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []

        obs = self.env.reset()
        obs = obs.astype(np.float32)
        if self.policy.rms:
            self.policy.rms.update(obs)
        for t in range(self._mb_size):
            mb_obs.append(obs)
            action = self.policy.select_action(torch.from_numpy(obs).to(device), stochastic=True)
            value = self.value_net.output_value(torch.from_numpy(obs).to(device))
            mb_actions.append(action)
            mb_values.append(value)
            obs, reward, done, _ = self.env.step(action)
            obs = obs.astype(np.float32)
            if self.policy.rms:
                self.policy.rms.update(obs)
            mb_rewards.append(reward)
            bool2float = lambda done: 1.0 if done else 0.0  # maybe bool_to_float
            mb_dones.append([bool2float(i) for i in done])

        mb_obs = np.asarray(mb_obs, dtype=np.float32)  # shape of mb_obs = (_mb_size, env_num, obs_space.shape[0])
        mb_actions = np.asarray(mb_actions, dtype=np.float32)  # shape of mb_actions = (_mb_size, env_num, action_space.shape[0])
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)  # shape of mb_rewards = (_mb_size, env_num, 1)
        mb_rewards = np.reshape(mb_rewards, (-1, mb_obs.shape[1], 1))
        mb_values = np.asarray(mb_values, dtype=np.float32)  # shape of mb_values = (_mb_size, env_num, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.float32)  # shape of mb_dones = (_mb_size, env_num, 1)
        mb_dones = np.reshape(mb_dones, (-1, mb_obs.shape[1], 1))
        last_value = self.value_net.output_value(torch.from_numpy(obs).to(device))  # shape of old last_value = (env_num, 1)
        last_value = np.reshape(last_value, (1, mb_values.shape[1], 1))  # shape of new last_value = (1, env_num, 1)

        seg = {
            "mb_obs": mb_obs,
            "mb_actions": mb_actions,
            "mb_rewards": mb_rewards,
            "mb_values": mb_values,
            "mb_dones": mb_dones,
            "last_value": last_value,
        }

        return seg



