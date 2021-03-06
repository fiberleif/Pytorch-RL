import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=(), clip_range=[-5.0, 5.0]):
        self._sum = torch.zeros(shape, dtype=torch.float32).to(device)
        self._sumsq = torch.ones(shape, dtype=torch.float32).to(device) * epsilon
        self._count = torch.ones((), dtype=torch.float32).to(device) * epsilon
        self.shape = shape
        self._clip_range = clip_range  # for observation normalization
        self._update_mean_and_std()  # compute mean & std of observation

    def normalize(self, x):
        return torch.clamp((x - self._mean) / self._std, min(self._clip_range), max(self._clip_range))

    def update(self, x):
        newsum = torch.tensor(x.sum(axis=0).ravel().reshape(self.shape), dtype=torch.float32).to(device)
        newsumsq = torch.tensor(np.square(x).sum(axis=0).ravel().reshape(self.shape), dtype=torch.float32).to(device)
        newcount = torch.tensor(len(x), dtype=torch.float32).to(device)

        self._sum += newsum
        self._sumsq += newsumsq
        self._count += newcount
        self._update_mean_and_std()

    def _update_mean_and_std(self):
        self._mean = self._sum / self._count
        self._std = torch.sqrt(torch.max(self._sumsq / self._count - self._mean ** 2,
                                         1e-2 * torch.ones_like(self._sumsq, dtype=torch.float32)))
