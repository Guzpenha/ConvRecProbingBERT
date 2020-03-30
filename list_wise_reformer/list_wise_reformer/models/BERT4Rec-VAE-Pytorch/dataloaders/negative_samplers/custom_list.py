from .base import AbstractNegativeSampler

from tqdm import trange
import numpy as np
from IPython import embed


class CustomListSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'customlist'

    def generate_negative_samples(self):
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        # the sampled items are already in the
        # pickled data on the validation items
        for user in trange(self.user_count):
            negative_samples[user] = self.test[user][1:]
        return negative_samples