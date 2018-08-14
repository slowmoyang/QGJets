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
                 cyclic=False):

        self._dataset = dataset
        self._batch_size = batch_size
        self._cyclic = cyclic

        if cyclic:
            self._next = self._cyclic_next
        else:
            self._next = self._non_cyclic_next

        self._num_examples = len(self._dataset)

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
            batch = self._next()
            return batch

    def __iter__(self):
        self._start = 0
        return self




def check_batch_time(data_iter, num_iter=10):
    elapsed_times = []
    for _ in range(num_iter):
        start_time = time.time()
        data_iter.next()
        elapsed_times.append(time.time() - start_time)
    elapsed_times = np.array(elapsed_times)
    print("[Elapsed time] mean: {mean:.5f} / stddev {stddev:5f}".format(
        mean=elapsed_times.mean(),
        stddev=elapsed_times.std()))
    print("Batch Size: {}".format(data_iter.batch_size))
    print("Iteration: {}".format(num_iter))
    return elapsed_times