from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ROOT

from keras4hep.data import BaseTreeDataset
from keras4hep.data import DataIterator


class HybridSet(BaseTreeDataset):
    def __init__(self, path, seq_maxlen=None, extra=None, tree_name="jetAnalyser"):
        """
        Arguments
          - path: A str. A path to a root file.
          - seq_maxlen: A dict.
          - extra: A list of strings.
          - tree_name: A str.
        """
        keys = ["x_img", "x_kin", "x_pid", "x_len", "y"]
        if extra is not None:
            keys += extra

        if seq_maxlen is None:
            seq_maxlen = {
                "x_kin": 30, 
                "x_pid": 30
            }

        super(HybridSet, self).__init__(
            path=path,
            tree_name=tree_name,
            keys=keys,
            seq_maxlen=seq_maxlen)

        self._extra = extra
        self._channels = [
            "image_chad_pt_33", "image_chad_mult_33",
            "image_nhad_pt_33", "image_nhad_mult_33",
            "image_photon_pt_33", "image_photon_mult_33",
            "image_electron_pt_33", "image_electron_mult_33",
            "image_muon_pt_33", "image_muon_mult_33"
        ]

        # 0 means neutral hadorn
        pid_list = [-11, 11, -13,  13, 22, 0]
        self._pid_map = {pid: idx for idx, pid in enumerate(pid_list, 2)}

        chad_pid = [-2212, -321, -211, 211, 321, 2212]
        self._pid_map.update({pid: 0 if pid < 0 else 1 for pid in chad_pid})

        self._convert_pid_to_idx = np.vectorize(lambda pid: self._pid_map[pid])

        self._embedding_input_dim = 7 + 1



    def _get_example(self, idx):
        self._tree.GetEntry(idx)

        x_img = [np.float32(getattr(self._tree, each)) for each in self._channels]
        x_img = np.stack(x_img)
        x_img = x_img.reshape(-1, 33, 33)

        # Constituents
        pt = np.array(self._tree.dau_pt, dtype=np.float32)
        deta = np.array(self._tree.dau_deta, dtype=np.float32)
        dphi = np.array(self._tree.dau_dphi, dtype=np.float32)

        pid = np.array(self._tree.dau_pid, dtype=np.int64)
        x_pid = self._convert_pid_to_idx(pid)

        x_kin = [
            pt,
            deta,
            dphi,
        ]
        x_kin = np.stack(x_kin, axis=-1)

        # From highest to lowest 
        pt_order = np.argsort(pt)[::-1]
        x_kin = x_kin[pt_order]

        seqlen = len(pt)
        if seqlen > self._seq_maxlen["x_kin"]:
            seqlen = self._seq_maxlen["x_kin"]
        x_len = np.int64(seqlen - 1)

        y = [0, 1] if np.int64(self._tree.label) else [1, 0]

        example = {
            "x_img": x_img,
            "x_kin": x_kin,
            "x_pid": x_pid,
            "x_len": x_len,
            "y": y
        }

        if self._extra is not None:
            extra = {each: getattr(self._tree, each) for each in self._extra}
            example.update(extra)
        
        return example


def get_data_iter(path,
                  extra=[],
                  batch_size=128,
                  **kwargs):
    dset = HybridSet(path=path, extra=extra)
    data_iter = DataIterator(dset, batch_size=batch_size, **kwargs)
    return data_iter


def _test():
    from keras4hep.projects.qgjets.utils import get_dataset_paths
    from dataset import get_data_iter

    paths = get_dataset_paths(min_pt=100)
    data_iter = get_data_iter(paths["training"])


    batch = data_iter.next()
    for key, value in batch.iteritems():
        print(key, value.shape)

    print(data_iter._dataset[:1]["x_img"].shape[1:])

    print(data_iter.get_shape("x_img", False))
    print(data_iter.get_shape("x_img", True))

if __name__ == "__main__":
    _test()
