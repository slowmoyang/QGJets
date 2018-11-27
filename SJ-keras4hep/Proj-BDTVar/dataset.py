from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ROOT

from keras4hep.data import BaseTreeDataset
from keras4hep.data import DataIterator


class BDTVarSet(BaseTreeDataset):
    def __init__(self, path, extra=None, tree_name="jetAnalyser"):
        """
        Arguments
          - path: A str. A path to a root file.
          - seq_maxlen: A dict.
          - extra: A list of strings.
          - tree_name: A str.
        """
        keys = ["x", "y"]
        if extra is not None:
            keys += extra

        super(BDTVarSet, self).__init__(
            path=path,
            tree_name=tree_name,
            keys=keys)

        self._extra = extra
        self._features = ["pt", "eta", "ptd", "major_axis", "minor_axis",
                          "chad_mult", "nhad_mult", "photon_mult",
                          "electron_mult", "muon_mult"]

    def _get_example(self, idx):
        self._tree.GetEntry(idx)

        x = [getattr(self._tree, each) for each in self._features]
        x = np.array(x, dtype=np.float32)
        
        y = [0, 1] if self._tree.label else [1, 0]

        example = {
            "x": x,
            "y": y
        }

        if self._extra is not None:
            extra = {each: getattr(self._tree, each) for each in self._extra}
            example.update(extra)
        
        return example


def get_data_iter(path,
                  batch_size=128,
                  extra=[],
                  **kwargs):
    dset = BDTVarSet(path=path, extra=extra)
    data_iter = DataIterator(dset, batch_size=batch_size, **kwargs)
    return data_iter


def _test():
    path = "/store/slowmoyang/QGJets/dijet_100_110/dijet_100_110_test.root"
    dset = BDTVarSet(path, extra=["pt", "eta"])
    data_iter = DataIterator(dset, batch_size=128)

    batch = data_iter.next()
    for key, value in batch.iteritems():
        print(key, value.shape)

    print(data_iter._dataset[:1]["x"].shape[1:])

    print(data_iter.get_shape("x", False))
    print(data_iter.get_shape("x", True))

if __name__ == "__main__":
    _test()
