from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

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
