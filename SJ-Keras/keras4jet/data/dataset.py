from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ROOT

from keras4jet.data.batch import Batch


class BaseTreeDataset(object):
    def __init__(self, path, tree_name, keys):
        self._root_file = ROOT.TFile.Open(path, "READ")
        self._tree = self._root_file.Get(tree_name)
        self._keys = keys

        self._path = path
        self._tree_name = tree_name

    def __len__(self):
        return int(self._tree.GetEntries())

    def _get_example(self, idx):
        raise NotImplementedError

    def __getitem__(self, key):
        if isinstance(key, int):
            # TODO negative(?) indxing like data_loader[-1]
            if key < 0 or key >= len(self):
                raise IndexError
            return self._get_example(key)
        elif isinstance(key, slice):

            batch = {each: [] for each in self._keys}

            for idx in range(*key.indices(len(self))):
                example = self._get_example(idx)

                for key in self._keys:
                    batch[key].append(example[key])

            batch = Batch({key: np.array(value) for key, value in batch.items()})
            return batch
        else:
            raise TypeError


