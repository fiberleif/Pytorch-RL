import numpy as np

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py


class ReplayBuffer(object):
    """ Replay Buffer to store transitions"""
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, observation, action, reward, next_observation, done):
        # add tuple of (observation, action, reward, next_observation, done)
        data = (observation, action, reward, next_observation, done)
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size=64):
        # sample tuples of (observation, action, reward, next_observation, done)
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            r.append(np.array(R, copy=False))
            u.append(np.array(U, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(r).reshape(-1, 1), np.array(u), np.array(d).reshape(-1, 1)
