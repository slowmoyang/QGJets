from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import ROOT

from keras4hep.data import BaseTreeDataset
from keras4hep.data import DataIterator


class JetSeqSet(BaseTreeDataset):
    def __init__(self, path, seq_maxlen, extra=None, tree_name="jetAnalyser"):
        """
        Arguments
          - path: A str. A path to a root file.
          - seq_maxlen: A dict.
          - extra: A list of strings.
          - tree_name: A str.
        """
        keys = ["x", "x_len", "y"]
        if extra is not None:
            keys += extra

        super(JetSeqSet, self).__init__(
            path=path,
            tree_name=tree_name,
            keys=keys,
            seq_maxlen=seq_maxlen)

        self._extra = extra

    def _get_example(self, idx):
        self._tree.GetEntry(idx)

        # Constituents
        pt = np.array(self._tree.dau_pt, dtype=np.float32)
        deta = np.array(self._tree.dau_deta, dtype=np.float32)
        dphi = np.array(self._tree.dau_dphi, dtype=np.float32)
        charge = np.array(self._tree.dau_charge, dtype=np.float32)

        pid = np.int64(self._tree.dau_pid)
        is_photon = np.where(pid == 22, 1, 0)
        is_electron = np.where(np.abs(pid) == 11, 1, 0)
        is_muon = np.where(np.abs(pid) == 13, 1, 0)
        is_nhad = np.where(pid == 0, 1, 0)

        is_not_chad = is_photon + is_electron + is_muon + is_nhad
        is_chad = np.logical_not(is_not_chad)

        x = [
            pt,
            deta,
            dphi,
            charge,
            is_chad,
            is_electron,
            is_muon,
            is_photon,
            is_nhad
        ]
        x = np.stack(x, axis=-1)

        # From highest to lowest 
        pt_order = np.argsort(pt)[::-1]
        x = x[pt_order]

        seqlen = len(pt)
        if seqlen > self._seq_maxlen["x"]:
            seqlen = self._seq_maxlen["x"]
        x_len = np.int64(seqlen - 1)
 
        y = [0, 1] if np.int64(self._tree.label) else [1, 0]

        example = {
            "x": x,
            "x_len": x_len,
            "y": y
        }

        if self._extra is not None:
            extra = {each: getattr(self._tree, each) for each in self._extra}
            example.update(extra)
        
        return example

def get_data_iter(path,
                  seq_maxlen,
                  extra=[],
                  batch_size=128,
                  **kwargs):
    dset = JetSeqSet(path=path, seq_maxlen=seq_maxlen, extra=extra)
    data_iter = DataIterator(dset, batch_size=batch_size, **kwargs)
    return data_iter

def _test():
    path = "/store/slowmoyang/QGJets/dijet_100_110/dijet_100_110_test.root"
    dset = JetSeqSet(path, extra=["pt", "eta"],
                     seq_maxlen={"x": 50})
    data_iter = DataIterator(dset, batch_size=128)


    batch = data_iter.next()
    for key, value in batch.iteritems():
        print(key, value.shape)

    print(data_iter.get_shape("x", False))
    print(data_iter.get_shape("x", True))

    data_iter.fit_generator_input = {
        "x": ["x"],
        "y": ["y"]
    }
    data_iter.fit_generator_mode = True
    for idx, (x, y) in enumerate(data_iter):
        if idx == 3:
            break

        print(len(x))
        for each in x:
            print("x: {}".format(each.shape))




if __name__ == "__main__":
    _test()
