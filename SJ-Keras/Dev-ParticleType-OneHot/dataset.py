from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

import numpy as np
import ROOT

from keras4jet.data.dataset import BaseTreeDataset
from keras4jet.data.data_iter import DataIterator


class ParticleTypeOneHotDataset(BaseTreeDataset):
    def __init__(self, path, seq_maxlen, tree_name="jetAnalyser"):
        super(ParticleTypeOneHotDataset, self).__init__(
            path=path,
            tree_name=tree_name,
            keys=["x", "y"],
            seq_maxlen=seq_maxlen)

    def _get_example(self, idx):
        self._tree.GetEntry(idx)

        # Constituents
        pt = np.array(self._tree.dau_pt, dtype=np.float32)
        deta = np.array(self._tree.dau_deta, dtype=np.float32)
        dphi = np.array(self._tree.dau_dphi, dtype=np.float32)

        charge = np.array(self._tree.dau_charge, dtype=np.float32)
        is_neutral = np.where(charge == 0, 1.0, 0.0).astype(np.float32)

        pid = np.array(self._tree.dau_pid, dtype=np.int64) 

        is_electron = np.where(np.abs(pid) == 11, 1, 0)
        is_muon = np.where(np.abs(pid) == 13, 1, 0)
        is_neutral_hadron = (pid == 0)
        is_photon = (pid == 22)

        is_not_charged_hadron = (is_electron + is_muon + is_neutral_hadron + is_photon).astype(bool)
        is_charged_hadron = np.bitwise_not(is_not_charged_hadron)

        x = np.hstack((
            pt,
            deta,
            dphi,
            charge,
            is_neutral,
            is_charged_hadron,
            is_electron,
            is_muon,
            is_photon,
            is_neutral_hadron,
        ))

        # From highest to lowest 
        pt_order = np.argsort(pt)[::-1]
        x = x[pt_order]
        
        y = 0 if self._tree.label == 1 else 1

        example = {
            "x": x,
            "y": y
        }
        
        return example


if __name__ == "__main__":
    path = "/store/slowmoyang/QGJets/data/root_100_200/2-Refined/dijet_test_set.root"
    dset = ParticleTypeOneHotDataset(path, seq_maxlen={"x": 30})
    data_iter = DataIterator(dset, batch_size=128)

    batch = data_iter.next()
    for key, value in batch.iteritems():
        print(key, value.shape)
