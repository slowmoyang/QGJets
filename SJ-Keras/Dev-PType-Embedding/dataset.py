from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections import defaultdict
import numpy as np
import ROOT

import sys
sys.path.append("..")
from keras4jet.data.dataset import BaseTreeDataset
from keras4jet.data.data_iter import DataIterator

# default = charged hadrons
_PID_MAP = defaultdict(lambda: 1)
# neutral hadron
_PID_MAP[0] = 0
# electron
_PID_MAP[11] = 2
_PID_MAP[-11] = 2
# muon
_PID_MAP[13] = 3
_PID_MAP[-13] = 3
# photon
_PID_MAP[22] = 4

_convert_pdgid_to_index = np.vectorize(lambda pdgid: _PID_MAP[pdgid])

class PTypeDataset(BaseTreeDataset):
    def __init__(self, path, seq_maxlen, extra=None, tree_name="jetAnalyser"):
        """
        Arguments
          - path: A str. A path to a root file.
          - seq_maxlen: A dict.
          - extra: A list of strings.
          - tree_name: A str.
        """
        self._extra = extra
        keys = ["x_kinematics", "x_pid", "y"]
        if extra is not None:
            keys += extra

        super(PTypeDataset, self).__init__(
            path=path,
            tree_name=tree_name,
            keys=keys,
            seq_maxlen=seq_maxlen)

    def _get_example(self, idx):
        self._tree.GetEntry(idx)

        # Constituents
        pt = np.array(self._tree.dau_pt, dtype=np.float32)
        deta = np.array(self._tree.dau_deta, dtype=np.float32)
        dphi = np.array(self._tree.dau_dphi, dtype=np.float32)

        charge = np.array(self._tree.dau_charge, dtype=np.float32)
        is_neutral = np.where(charge == 0, 1.0, 0.0).astype(np.float32)

        x_kinematics = np.stack(
            arrays=[
                pt,
                deta,
                dphi,
                charge,
                is_neutral],
            axis=-1)

        x_pid = np.array(self._tree.dau_pid, dtype=np.int64) 
        x_pid = _convert_pdgid_to_index(x_pid)

        # From highest to lowest 
        pt_order = np.argsort(pt)[::-1]
        x_kinematics = x_kinematics[pt_order]
        x_pid = x_pid[pt_order]
        
        y = 0 if self._tree.label == 1 else 1

        example = {
            "x_kinematics": x_kinematics,
            "x_pid": x_pid,
            "y": y
        }

        if self._extra is not None:
            extra = {each: getattr(self._tree, each) for each in self._extra}
            example.update(extra)
        
        return example

def get_data_iter(path, seq_maxlen, batch_size, cyclic=False, extra=[]):
    dset = PTypeDataset(path=path, seq_maxlen=seq_maxlen, extra=extra)
    data_iter = DataIterator(dset, batch_size=batch_size, cyclic=cyclic)
    return data_iter

def _test():
    path = "/store/slowmoyang/QGJets/data/root_100_200/2-Refined/dijet_test_set.root"
    dset = PTypeDataset(
        path,
        seq_maxlen={
            "x_kinematics": 30,
            "x_pid": 30
        },
        extra=["pt", "eta"])
    data_iter = DataIterator(dset, batch_size=128)

    batch = data_iter.next()
    for key, value in batch.iteritems():
        print(key, value.shape)

    print(data_iter._dataset[:1]["x_kinematics"].shape[1:])
    print(data_iter._dataset[:1]["x_pid"].shape[1:])

    print(data_iter.get_shape("x_kinematics", False))
    print(data_iter.get_shape("x_pid", True))

if __name__ == "__main__":
    _test()
