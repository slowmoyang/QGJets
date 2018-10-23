from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import time

import numpy as np

import ROOT


class DataIterator(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 cyclic=False,
                 shuffle=False):

        self._dataset = dataset
        self._batch_size = batch_size
        self._cyclic = cyclic
        self._shuffle = shuffle


        self._num_examples = len(self._dataset)

        if cyclic:
            self._next = self._cyclic_next
        else:
            if shuffle:
                self._indices = self._get_shuffled_indices()
                self._next = self._non_cyclic_shuffle_next
            else:
                self._next = self._non_cyclic_next


        self._start = 0

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        if size <= 0:
            raise ValueError
        self._batch_size = size

    def __len__(self):
        # TODO
        if self._cyclic:
            warnings.warn(
                "cyclic mode... length is # of batches per a epoch",
                Warning)
        num_batches = int(np.ceil(self._num_examples / self._batch_size))
        return num_batches

    def __getitem__(self, key):
        return self._dataset[key]

    def __next__(self):
        return self._next()

    # NOTE python2 support
    next = __next__

    def _non_cyclic_next(self):
        if self._start + 1 >= self._num_examples:
            raise StopIteration

        end = self._start + self._batch_size
        slicing = slice(self._start, end)
        self._start = end

        batch = self[slicing]
        return batch

    def _non_cyclic_shuffle_next(self):
        if self._start + 1 >= self._num_examples:
            raise StopIteration

        end = self._start + self._batch_size
        slicing = [self._indices.pop() for _ in range(self._start, end)]
        self._start = end

        batch = self[slicing]
        return batch

    def _cyclic_next(self):
        if self._start + 1 < len(self):
            end = self._start + self._batch_size
            slicing = slice(self._start, end)

            if end <= self._num_examples:
                self._start = end
                batch = self[slicing]
                return batch
            else:
                batch = self[slicing]
                self._start = 0
                end = end - len(self)
                batch1 = self[slice(self._start, end)]
                self._start = end
                batch += batch1
                return batch
        else:
            self._start = 0
            end = self._start + self._batch_size
            batch = self[slice(self._start, end)]
            self._start = end
            return batch

    def __iter__(self):
        self._start = 0
        self._indices = self._get_shuffled_indices()
        return self

    def _get_shuffled_indices(self):
        indices = np.arange(self._num_examples)
        np.random.shuffle(indices)
        indices = list(indices)
        return indices

    def get_shape(self, key, batch_shape=False):
        shape = self._dataset[:1][key].shape[1:]
        if batch_shape:
            return (self._batch_size, ) + shape
        else:
            return shape
        
