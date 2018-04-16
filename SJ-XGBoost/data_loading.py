from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def load_dataset(path, x, y="label", extra=None):
    npz_file = np.load(path)

    design_matrix = [npz_file[key] for key in x]
    design_matrix = np.array(design_matrix, dtype=np.float32)
    design_matrix = design_matrix.T

    label_vector = npz_file[y]

    dataset = (design_matrix, label_vector)

    if extra is None:
        return dataset
    else:
        extra = {key: npz_file[key] for key in extra}
        return (dataset, extra) 

