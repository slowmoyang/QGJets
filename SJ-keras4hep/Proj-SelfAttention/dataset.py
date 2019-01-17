from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import itertools

import numpy as np
import ROOT

from keras4hep.data import BaseTreeDataset
from keras4hep.data import DataIterator


class PIDOneHotSet(BaseTreeDataset):
    def __init__(self, path, seq_maxlen, extra=None, tree_name="jetAnalyser"):
        self.fit_generator_input = {
            "x": ["x"] ,
            "y": ["y"]
        }

        keys = list(itertools.chain.from_iterable(self.fit_generator_input.values()))
        if extra is not None:
            keys += extra

        super(PIDOneHotSet, self).__init__(
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

        pid = np.array(self._tree.dau_pid, dtype=np.int64)
        is_nhad = (pid == 0)
        is_photon = (pid == 22)
        is_electron = np.abs(pid) == 11 
        is_muon = np.abs(pid) == 13

        is_not_chad = sum([is_nhad, is_photon, is_electron, is_muon])
        is_chad = np.logical_not(is_not_chad)        

        x = [
            pt,
            deta,
            dphi,
            charge,
            is_chad,
            is_nhad,
            is_photon,
            is_electron,
            is_muon
        ]
        x = np.stack(x, axis=-1)

        # From highest to lowest 
        # pt_order = np.argsort(pt)[::-1]
        # x = x[pt_order]

 
        y = [0, 1] if np.int64(self._tree.label) else [1, 0]

        example = {
            "x": x,
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
    dset = PIDOneHotSet(path=path, seq_maxlen=seq_maxlen, extra=extra)
    data_iter = DataIterator(dset, batch_size=batch_size, **kwargs)
    return data_iter

def _test():
    from keras4hep.projects.qgjets.utils import get_dataset_paths
    path = get_dataset_paths(min_pt=100)["training"]

    seq_maxlen = {
        "x": (40, "float32"),
    }

    data_iter = get_data_iter(
        path=path,
        seq_maxlen=seq_maxlen,
        batch_size=2)

    batch = data_iter.next()




if __name__ == "__main__":
    _test()
