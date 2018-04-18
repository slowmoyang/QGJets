from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def load_dataset(path, features, label="label", extra=None):
    npz_file = np.load(path)

    design_matrix = [npz_file[each] for each in features]
    design_matrix = [each.astype(np.float32) for each in design_matrix]
    design_matrix = np.column_stack(design_matrix)

    feature_names = []
    for key in features:
        shape = npz_file[key].shape
        if len(shape) == 1:
            feature_names.append(key)
        elif len(shape) == 2:
            names = ["{}_{}".format(key, i) for i in range(shape[1])]
            feature_names += names
        else:
            raise ValueError(":p")

    label_vector = npz_file[label]

    dataset = (design_matrix, label_vector, feature_names)

    if extra is None:
        return dataset
    else:
        extra = {key: npz_file[key] for key in extra}
        return (dataset, extra) 
