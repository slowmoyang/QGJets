from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ROOT

from keras4hep.data import BaseTreeDataset
from keras4hep.data import DataIterator


class C10Set(BaseTreeDataset):
    def __init__(self, path, prep_path=None, extra=None, tree_name="jetAnalyser"):
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

        super(C10Set, self).__init__(
            path=path,
            tree_name=tree_name,
            keys=keys)

        self._extra = extra
        self._channels = [
            "image_chad_pt_33", "image_chad_mult_33",
            "image_nhad_pt_33", "image_nhad_mult_33",
            "image_photon_pt_33", "image_photon_mult_33",
            "image_electron_pt_33", "image_electron_mult_33",
            "image_muon_pt_33", "image_muon_mult_33"
        ]

        ############
        # for preprocessing
        ##############
        if prep_path is None:
            self._prep = False
        else:
            self._prep = True
            self._prep_path = prep_path
            self._prep_file = np.load(prep_path)

            mu = [self._prep_file["mu_" + each] for each in self._channels]
            mu = np.stack(mu)
            self._mu = mu.reshape(10, 33, 33)

            # A value of r = 10^{-5} was used to supress noise.
            # (arXiv:1612.01551 [hep-ph])
            sigma = [self._prep_file["sigma_" + each] for each in self._channels]
            sigma = np.stack(sigma)
            self._sigma = sigma.reshape(10, 33, 33) + 1e-5



    def _get_example(self, idx):
        self._tree.GetEntry(idx)

        x = [np.float32(getattr(self._tree, each)) for each in self._channels]
        x = np.stack(x)
        x = x.reshape(-1, 33, 33)

        # TODO move to iterator??
        if self._prep:
            normalizer = x.sum(axis=(1, 2), keepdims=True)
            normalizer[normalizer == 0] += 1e-6
            x /= normalizer
            x -= self._mu
            x /= self._sigma
        
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
                  prep_path=None,
                  extra=[],
                  batch_size=128,
                  **kwargs):
    dset = C10Set(path=path, prep_path=prep_path, extra=extra)
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

    print(data_iter._dataset[:1]["x"].shape[1:])

    print(data_iter.get_shape("x", False))
    print(data_iter.get_shape("x", True))

if __name__ == "__main__":
    _test()
