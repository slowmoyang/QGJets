from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
import numpy as np

from .base import BaseDataLoader

class FeatureLoader(BaseDataLoader):
    __slots__ = ("root_file", "tree", "path", "example_list", "extra",
                 "num_classes", "batch_size", "cyclic", "tree_name",
                 "keys", "get_data", "_start", "x", "y")
    def __init__(self,
                 path,
                 x,
                 batch_size,
                 cyclic=True,
                 extra=[],
                 y="label",
                 num_classes=2,
                 tree_name="jetAnalyser"):

        example_list = ["x", "y"]

        super(FeatureLoader, self).__init__(
            path, example_list, extra, num_classes, batch_size, cyclic, tree_name)

        self.x = x
        self.y = y


    def _get_data(self, idx):
        self.tree.GetEntry(idx)
        example = dict()

        example["x"] = np.array(
            object=[getattr(self.tree, each) for each in self.x],
            dtype=np.float32)

        example["y"] = np.zeros(self.num_classes, dtype=np.int64)
        example["y"][int(getattr(self.tree, self.y))] = 1

        return example

