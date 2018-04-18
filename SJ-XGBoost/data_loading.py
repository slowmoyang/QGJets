from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def load_dataset(path, features, label="label", extra=None):
    npz_file = np.load(path)

    design_matrix = [npz_file[key] for key in features]
    design_matrix = np.column_stack(design_matrix)

    label_vector = npz_file[label]

    dataset = (design_matrix, label_vector)

    if extra is None:
        return dataset
    else:
        extra = {key: npz_file[key] for key in extra}
        return (dataset, extra) 
