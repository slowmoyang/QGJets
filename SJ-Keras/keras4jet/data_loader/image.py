from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
import numpy as np

import warnings

from .base import DataLoaderBase


class ImageSetLoader(DataLoaderBase):
    __slots__ = ("root_file", "tree", "path", "example_list", "extra",
                 "num_classes", "batch_size", "cyclic", "tree_name",
                 "keys", "get_data", "_start", "x", "x_shape", "y")
    def __init__(self,
                 path,
                 x,
                 x_shape,
                 batch_size,
                 cyclic=True,
                 y="label",
                 num_classes=2,
                 extra=[],
                 tree_name="jetAnalyser"):

        example_list = ["x", "y"]

        super(ImageSetLoader, self).__init__(
            path, example_list, extra, num_classes, batch_size, cyclic, tree_name)

        self.x = x
        self.x_shape = x_shape
        self.y = y

    def _get_data(self, idx):
        self.tree.GetEntry(idx)
        example = dict()

        example["x"] = np.array(
            object=[getattr(self.tree, each) for each in self.x],
            dtype=np.float32).reshape(self.x_shape)

        example["y"] = np.zeros(self.num_classes, dtype=np.int64)
        example["y"][int(getattr(self.tree, self.y))] = 1

        return example

