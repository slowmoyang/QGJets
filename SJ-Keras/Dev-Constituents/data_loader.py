import sys
sys.path.append("..")

from keras4jet.data_loader.base import BaseMultiSeqLoader
from keras4jet.data_loader.base import Batch

import numpy as np
import ROOT
from collections import OrderedDict
from collections import Sequence

class ConstituentsLoader(BaseMultiSeqLoader):
    def __init__(self,
                 path,
                 seq_maxlen=None,
                 batch_size=512,
                 cyclic=False,
                 extra=[],
                 label="label",
                 num_classes=2,
                 tree_name="jetAnalyser",
                 padding="pre",
                 truncating="pre"):
        
        features_list = ["x_jet", "x_constituents"]

        self._x_jet = ["major_axis", "minor_axis", "cmult", "nmult", "ptD"]

        if isinstance(seq_maxlen, dict):
            pass 
        elif seq_maxlen is None:
            root_file = ROOT.TFile.Open(path, "READ")
            tree = root_file.jetAnalyser
            n_dau = np.array([entry.n_dau for entry in tree])
            root_file.Close()
            del root_file, tree
            maxlen = int(n_dau.mean() + 2 * n_dau.std())
            seq_maxlen = {
                "x_constituents": maxlen,
            }
        elif isinstance(seq_maxlen, Sequence) and len(seq_maxlen) == 1:
            seq_maxlen = {
                "x_constituents": seq_maxlen[0],
            } 
        else:
            raise ValueError
        
        super(ConstituentsLoader, self).__init__(
            path, features_list, label, extra, seq_maxlen, batch_size, cyclic,
            num_classes, tree_name, padding, truncating)
        
    def _get_data(self, idx):
        self.tree.GetEntry(idx)
        example = dict()
        
        # Daughters
        dau_p4 = np.array([[p[mu] for mu in range(4)] for p in self.tree.dau_p4], dtype=np.float32)
        dau_charge = np.array(self.tree.dau_charge, dtype=np.float32)
        dau_is_neutral = np.where(dau_charge == 0, 1, 0).astype(np.float32)

        dau_charge = np.expand_dims(dau_charge, axis=-1)
        dau_is_neutral = np.expand_dims(dau_is_neutral, axis=-1)

        x_constituents = np.hstack(
            tup=(dau_p4,
                 dau_charge,
                 dau_is_neutral))

        # From highest to lowest 
        energy_order = np.argsort(dau_p4[:, -1])[::-1]
        example["x_constituents"] = x_constituents[energy_order]
        
        # Global
        example["x_jet"] = np.array(
            [getattr(self.tree, each) for each in self._x_jet],
            dtype=np.float32)
        
                # Label
        example["label"] = np.zeros(self._num_classes, dtype=np.int64)
        example["label"][int(getattr(self.tree, self._label))] = 1
        
        return example

    def get_shapes(self, batch_shape=False, as_kwargs=False):
        example = self[0]
        shapes = OrderedDict({key: example[key].shape for key in self._features_list})
        if as_kwargs:
            for key in self._features_list:
                shapes[key + "_shape"] = shapes.pop(key)
        if batch_shape:
            for key in shapes:
                shapes[key] = (None, ) + shapes.pop(key)
        return shapes


if __name__ == "__main__":
    loader = ConstituentsLoader("/data/slowmoyang/QGJets/root_100_200/2-Refined/dijet_training_set.root")
    batch = loader.next()
    for key, value in batch.iteritems():
        print("{}: {}".format(key, value.shape))
