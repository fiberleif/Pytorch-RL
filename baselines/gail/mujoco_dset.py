'''
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
'''

from baselines import logger
import numpy as np


class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels


class Mujoco_Dset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, data_subsample_freq=1, randomize=True):
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        obs = traj_data['obs'][:traj_limitation]
        acs = traj_data['acs'][:traj_limitation]
        traj_lens = traj_data['ep_lens'][:traj_limitation]
        self.rets = traj_rets = traj_data['ep_rets'][:traj_limitation]

        print('Expert dataset size: {0} transitions ({1} trajectories)'.format(traj_lens.sum(), len(traj_lens)))
        print('Expert average return:', traj_rets.mean())

        # Subsample trajs
        start_times = np.random.randint(0, data_subsample_freq, size=obs.shape[0])
        self.obs = np.concatenate([obs[i, start_times[i]:l:data_subsample_freq, :]
                                         for i, l in enumerate(traj_lens)], axis=0)
        self.acs = np.concatenate([acs[i, start_times[i]:l:data_subsample_freq, :]
                                         for i, l in enumerate(traj_lens)], axis=0)
        stacked = np.concatenate(
        [np.arange(start_times[i], l, step=data_subsample_freq) for i, l in enumerate(traj_lens)]).astype(float)

        print('Subsample data every {0} timesteps'.format(data_subsample_freq))
        print('Final dataset size: {0} transitions (average {1} per traj)'.format(stacked.shape[0],
                                            float(stacked.shape[0])/traj_lens.shape[0]))

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))
        # if len(obs.shape[2:]) != 0:
        #     self.obs = np.reshape(obs, [-1, np.prod(subsample_obs.shape[2:])])
        #     self.acs = np.reshape(acs, [-1, np.prod(subsample_acs.shape[2:])])
        # else:
        #     self.obs = np.vstack(subsample_obs)
        #     self.acs = np.vstack(subsample_acs)

        if len(self.acs) > 2:
            self.acs = np.squeeze(self.acs)
        assert len(self.obs) == len(self.acs)
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.randomize)

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

    def plot(self):
        import matplotlib.pyplot as plt
        plt.hist(self.rets)
        plt.savefig("histogram_rets.png")
        plt.close()


def test(expert_path, traj_limitation, plot):
    dset = Mujoco_Dset(expert_path, traj_limitation=traj_limitation)
    if plot:
        dset.plot()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_path", type=str, default="../data/deterministic.trpo.Hopper.0.00.npz")
    parser.add_argument("--traj_limitation", type=int, default=None)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()
    test(args.expert_path, args.traj_limitation, args.plot)
