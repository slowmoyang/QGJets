from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Batch(object):
    def __init__(self, batch, **kwargs):
        batch = {key: np.array(value) for key, value in batch.items()}
        for key, value in batch.iteritems():
            setattr(self, key, value)

        self._batch = batch
        self._size = len(batch.values()[0])

    def __getitem__(self, key):
        return self._batch[key]

    def __len__(self):
        return self._size

    def __str__(self):
        summary = ""
        for key, value in self._batch.iteritems():
            summary += "{}: {}\n".format(key, value.shape)
        summary.strip("\n")
        return summary

    def __add__(self, other):
        concat = {key: np.concatenate([self[key], other[key]]) for key in self.keys()}
        concat = Batch(concat)
        return concat

    def keys(self):
        return self._batch.keys()

    def values(self):
        return self._batch.values()

    def iteritems(self):
        return self._batch.iteritems()

    def shuffle(self):
        shuffled_index = np.arange(len(self))
        for key, value in self.iteritems():
            self._batch[key] = value[shuffled_index]
