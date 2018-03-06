from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
import numpy as np

from keras.preprocessing.sequence import pad_sequences

import warnings


class DataLoader(object):

    def __init__(self,
                 path,
                 example_list,
                 extra,
                 num_classes,
                 batch_size,
                 cyclic,
                 tree_name="jetAnalyser"):

        self.root_file = ROOT.TFile.Open(path, "READ")
        self.tree = self.root_file.Get(tree_name)

        self.path = path
        self.example_list = example_list
        self.extra = extra
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.tree_name = tree_name

        self.keys = example_list + extra
        if len(self.extra) == 0:
            self.get_data = self._get_data
        else:
            self.get_data = self._get_data_with_extra
        self._start = 0
        
    def __len__(self):
        return int(self.tree.GetEntries())

    def _get_data(self, idx):
        raise NotImplementedError("")

    def _get_data_with_extra(self, idx):
        example = self._get_data(idx)
        for key in self.extra:
            example[key] = getattr(self.tree, key)
        return example

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError
            return self.get_data(key)
        elif isinstance(key, slice):
            batch = {key: [] for key in self.keys}

            for idx in xrange(*key.indices(len(self))):
                example = self.get_data(idx)

                for key in self.keys:
                    batch[key].append(example[key])

            batch = {key: np.array(value) for key, value in batch.items()}
            return batch
        else:
            raise TypeError
        
    def next(self):
        if self.cyclic:
            if self._start + 1 < len(self):
                end = self._start + self.batch_size
                slicing = slice(self._start, end)
                if end <= len(self):
                    self._start = end
                    return self[slicing]
                else:
                    batch = self[slicing]
                    
                    self._start = 0
                    end = end - len(self)

                    batch1 = self[slice(self._start, end)]
                    self._start = end
                    
                    batch = {key: np.append(batch[key], batch1[key], axis=0) for key in self.keys}
                    return batch
            else:
                self._start = 0
                return self.next()
        else:
            if self._start + 1 < len(self):
                end = self._start + self.batch_size
                slicing = slice(self._start, end)
                self._start = end
                return self[slicing]
            else:
                raise StopIteration
                
    def __next__(self):
        return self.next()

    def __iter__(self):
        for start in xrange(0, len(self), self.batch_size): 
            yield self[slice(start, start+self.batch_size)]


class FeaturesDataLoader(DataLoader):
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

        super(FeaturesDataLoader, self).__init__(
            path, example_list, extra, num_classes, batch_size, cyclic, tree_name)

        self.x = x
        self.y = y

        extra = set(self.keys).difference({"x", "y"})


    def _get_data(self, idx):
        self.tree.GetEntry(idx)
        example = dict()

        example["x"] = np.array(
            object=[getattr(self.tree, each) for each in self.x],
            dtype=np.float32)

        example["y"] = np.zeros(self.num_classes, dtype=np.int64)
        example["y"][int(getattr(self.tree, self.y))] = 1

        return example

    def _get_data_including_extra(self, idx):
        self.tree.GetEntry(idx)
        example = dict()

        example["x"] = np.array(
            object=[getattr(self.tree, each) for each in self.x],
            dtype=np.float32)

        example["y"] = np.zeros(self.num_classes, dtype=np.int64)
        example["y"][int(getattr(self.tree, self.y))] = 1


class ImageDataLoader(DataLoader):
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

        super(ImageDataLoader, self).__init__(
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


class SeqDataLoader(DataLoader):
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

        super(SeqDataLoader, self).__init__(
            path, example_list, extra, num_classes, batch_size, cyclic, tree_name)

        self.y = y

        max_n_dau = int(self.tree.GetMaximum("n_dau"))
        if maxlen is None:
            self.maxlen = max_n_dau
        elif isinstance(maxlen, int) and maxlen > 0:
            self.maxlen = maxlen
            if maxlen > max_n_dau:
                warnings.warn("maxlen({}) is larger than max 'n_dau' ({}) in data".format(
                    maxlen, max_n_dau))
        else:
            raise ValueError("maxlen")

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
            is_neutral
        ))
        x_daus = x_daus.T

        #
        pt_ordering = np.argsort(dau_pt)
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

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError
            example = self.get_data(key)
            # CMS AN-17-188.
            # Sec. 3.1 Slim jet DNN architecture (p. 11 / line 196-198)
            # When using recurrent networks the ordering is important, thus
            # our underlying assumption is that the most displaced (in case
            # of displacement) or the highest pT candidates matter the most.
            example["x_daus"] = np.expand_dims(example["x_daus"], axis=0)
            example["x_daus"] = pad_sequences(
                sequences=example["x_daus"],
                maxlen=self.maxlen,
                dtype=np.float32,
                padding="pre",
                truncating="pre",
                value=0.)
            example["x_daus"] = example["x_daus"].reshape(example["x_daus"].shape[1:])
            return example
        elif isinstance(key, slice):
            batch = {key: [] for key in self.keys}
            for idx in xrange(*key.indices(len(self))):
                example = self.get_data(idx)
                for key in self.keys:
                    batch[key].append(example[key])
            batch["x_daus"] = pad_sequences(
                sequences=batch["x_daus"],
                maxlen=self.maxlen,
                dtype=np.float32,
                padding="pre",
                truncating="pre",
                value=0.)
            batch = {key: np.array(value) for key, value in batch.items()}
            return batch
        else:
            raise TypeError
