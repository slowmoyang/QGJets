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
            self._mu = self._prep_file["mu"]
            # A value of r = 10^{-5} was used to supress noise.
            # (arXiv:1612.01551 [hep-ph])
            self._sigma = self._prep_file["sigma"] + 1e-5


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
        
        y = [0, 1] if self._tree.label == 1 else [1, 0]

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
                  batch_size=128,
                  extra=[],
                  **kwargs):
    dset = C10Set(path=path, prep_path=prep_path, extra=extra)
    data_iter = DataIterator(dset, batch_size=batch_size, **kwargs)
    return data_iter

def _test():
    path = "/store/slowmoyang/QGJets/dijet_100_110/dijet_100_110_test.root"
    prep_path = "./logs/dijet_100_110_training.npz"
    dset = C10Set(path, prep_path, extra=["pt", "eta"])
    data_iter = DataIterator(dset, batch_size=128)

    batch = data_iter.next()
    for key, value in batch.iteritems():
        print(key, value.shape)

    print(data_iter._dataset[:1]["x"].shape[1:])

    print(data_iter.get_shape("x", False))
    print(data_iter.get_shape("x", True))


def compute_mu_sigma(in_path):
    import os
    dset = C10Set(in_path)
    data_iter = DataIterator(dset, batch_size=512)

    print("MU")
    mu = []
    for idx, batch in enumerate(data_iter, 1):
        x = batch.x
        # normalization
        normalizer = x.sum(axis=(2, 3), keepdims=True)
        normalizer[normalizer == 0] += 1e-6
        x /= normalizer

        mu.append(x)
        print("\t[{}/{}] ({:.1f}) %".format(
            idx, len(data_iter), idx / len(data_iter) * 100))
    mu = np.concatenate(mu, axis=0)
    mu = mu.mean(axis=0)

    print("SIGMA")
    sigma = []
    for idx, batch in enumerate(data_iter, 1):
        x = batch.x

        # normalization
        normalizer = x.sum(axis=(2, 3), keepdims=True)
        normalizer[normalizer == 0] += 1e-6
        x /= normalizer 
        x -= mu
        sigma.append(x)
        print("\t[{}/{}] ({:.1f}) %".format(
            idx, len(data_iter), idx / len(data_iter) * 100))

    sigma = np.concatenate(sigma)
    sigma = sigma.std(axis=0)

    directory, basename = os.path.split(in_path)
    out_name = "preprocessing_" + basename.replace(".root", ".npz")

    out_path = os.path.join(directory, out_name)
    np.savez(out_path,
             mu=mu,
             sigma=sigma)



if __name__ == "__main__":
    format_str = "/store/slowmoyang/QGJets/dijet_{min_pt}_{max_pt}/dijet_{min_pt}_{max_pt}_training.root"
    for min_pt in [100, 200, 500, 1000]:
        max_pt = int(1.1 * min_pt)
        path = format_str.format(min_pt=min_pt, max_pt=max_pt)
        print(path)
        compute_mu_sigma(path)
