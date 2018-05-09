from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
import numpy as np

from .base import SeqDataLoaderBase

class AK4Loader(SeqDataLoaderBase):
    __slots__ = ("root_file", "tree", "path", "example_list", "extra",
                 "num_classes", "batch_size", "cyclic", "tree_name",
                 "keys", "get_data", "_start", "maxlen", "y")
    def __init__(self,
                 path,
                 batch_size,
                 maxlen=None,
                 cyclic=True,
                 extra=[],
                 y="label",
                 num_classes=2,
                 tree_name="jetAnalyser"):

        example_list = ["x_daus", "x_glob", "y"]

        super(AK4Loader, self).__init__(
            path, example_list, batch_size, maxlen, cyclic, extra, y,
            num_classes, tree_name)

    def _get_data(self, idx): 
        self.tree.GetEntry(idx)
        example = dict()

        # daughter features used as input
        dau_pt = np.array(self.tree.dau_pt, dtype=np.float32)
        dau_rel_pt = dau_pt / self.tree.pt
        dau_deta = np.array(self.tree.dau_deta, dtype=np.float32)
        dau_dphi = np.array(self.tree.dau_dphi, dtype=np.float32)
        dau_charge = np.array(self.tree.dau_charge, dtype=np.float32)
        #is_charged = np.where(dau_charge != 0, 0, 1)
        is_neutral = np.where(dau_charge == 0, 1, 0)

        #
        x_daus = np.vstack((
            dau_rel_pt,
            dau_deta,
            dau_dphi,
            dau_charge,
            is_neutral))
        x_daus = x_daus.T

        #
        pt_ordering = np.argsort(dau_pt)[::-1]
        example["x_daus"] = x_daus[pt_ordering]

        # x_glob: global input features
        example["x_glob"] = np.array([
            self.tree.pt,
            self.tree.eta,
            self.tree.phi,
            self.tree.n_dau])

        # Label
        example["y"] = np.zeros(self.num_classes, dtype=np.int64)
        example["y"][int(getattr(self.tree, self.y))] = 1

        return example

    def get_shape(self, batch_shape=False):
        shapes = [self[0][each].shape for each in ["x_daus", "x_glob"]] 
        if batch_shape:
            shapes = [tuple([None] + list(each)) for each in shapes]
        return shapes
