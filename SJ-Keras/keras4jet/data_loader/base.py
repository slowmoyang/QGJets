from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
import numpy as np

from keras.preprocessing.sequence import pad_sequences

import warnings

from collections import OrderedDict
#from collections import MutableMapping
import time


# TODO rearrange args

class Batch(object):
    def __init__(self, raw_batch, **kwargs):
        self._raw_batch = raw_batch

        for key, value in raw_batch.iteritems():
            setattr(self, key, value)

        for category, item_list in kwargs.iteritems():
            items = [raw_batch[key] for key in item_list]
            setattr(self, category, items)

        self._keys = set(raw_batch.keys())


    def __getitem__(self, key):
        return self._raw_batch[key]

    def __len__(self):
        pass

    def size(self):
        raise NotImplementedError

    def keys(self):
        return self._keys

    def iteritems(self):
        return self._raw_batch.iteritems()
        



class BaseDataLoader(object):
    def __init__(self,
                 path,
                 features_list,
                 label,
                 extra,
                 num_classes,
                 batch_size,
                 cyclic,
                 tree_name="jetAnalyser"):

        self.root_file = ROOT.TFile.Open(path, "READ")
        self.tree = self.root_file.Get(tree_name)

        self.path = path
        self._features_list = features_list
        self._label = label
        self._example_list = features_list + [label]
        self._extra = extra
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._cyclic = cyclic
        self._tree_name = tree_name

        self.keys = self._example_list + extra
        if len(self._extra) == 0:
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
        for key in self._extra:
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
        if self._cyclic:
            if self._start + 1 < len(self):
                end = self._start + self._batch_size
                slicing = slice(self._start, end)
                if end <= len(self):
                    self._start = end
                    batch = self[slicing]
                    return batch
                else:
                    batch = self[slicing]
                    self._start = 0
                    end = end - len(self)
                    batch1 = self[slice(self._start, end)]
                    self._start = end
                    batch = {key: np.append(batch[key], batch1[key], axis=0) for key in self.keys}
                    batch = Batch(batch, features=self._features_list, extra=self._extra)
                    return batch
            else:
                self._start = 0
                batch = self.next()
                return batch
        else:
            if self._start + 1 < len(self):
                end = self._start + self._batch_size
                slicing = slice(self._start, end)
                self._start = end
                batch = self[slicing]
                return batch
            else:
                raise StopIteration

    def __next__(self):
        return self.next()

    def __iter__(self):
        for start in xrange(0, len(self), self._batch_size): 
            yield self[slice(start, start + self._batch_size)]


    def check_batch_time(self, n=10):
        elapsed_times = []
        for _ in xrange(n):
            start_time = time.time()
            self.next()
            elapsed_times.append(time.time() - start_time)
        elapsed_times = np.array(elapsed_times)
        print("[Elapsed time] mean: {mean:.5f} / stddev {stddev:5f}".format(
            mean=elapsed_times.mean(),
            stddev=elapsed_times.std()))
        print("Batch Size: {}".format(self._batch_size))
        print("Iteration: {}".format(n))


class BaseSeqDataLoader(BaseDataLoader):
    def __init__(self,
                 path,
                 features_list,
                 batch_size,
                 maxlen,
                 cyclic,
                 extra,
                 y,
                 num_classes,
                 tree_name):

        super(BaseSeqDataLoader, self).__init__(
            path, features_list, y, extra, num_classes, batch_size, cyclic, tree_name)

        self.y = y

        self.maxlen = self.eval_maxlen(maxlen)

    def _get_data(self, idx): 
        raise NotImplementedError("")
        
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
            batch = Batch(batch, features=self._features_list, extra=self._extra)
            return batch
        else:
            raise TypeError

    def eval_maxlen(self, maxlen):
        if isinstance(maxlen, int) and maxlen > 0:
            max_n_dau = int(self.tree.GetMaximum("n_dau"))
            if maxlen > max_n_dau:
                warnings.warn("maxlen({}) is larger than max 'n_dau' ({}) in data".format(
                    maxlen, max_n_dau))
            output = maxlen
        elif isinstance(maxlen, str):
            n_dau = np.array([each.n_dau for each in self.tree])
            maxlen = maxlen.lower()
            if maxlen == "max":
                output = n_dau.max()
            elif maxlen == "mean":
                output = int(n_dau.mean())
            elif maxlen == "median":
                output = np.median(n_dau)
            else:
                raise ValueError
        elif isinstance(maxlen, float) and maxlen > 0 and maxlen < 1:
            n_dau = np.array(n_dau, dtype=np.int64)
            n_dau.sort()
            idx = int(len(n_dau)*maxlen)
            output = n_dau[idx]
        elif maxlen is None:
            output = int(self.tree.GetMaximum("n_dau"))
        else:
            raise ValueError

        return int(output)



class BaseMultiSeqLoader(BaseDataLoader):
    def __init__(self,
                 path,
                 features_list,
                 label,
                 extra,
                 seq_maxlen,
                 batch_size,
                 cyclic,
                 num_classes,
                 tree_name,
                 padding,
                 truncating):
        super(BaseMultiSeqLoader, self).__init__(
            path, features_list, label, extra, num_classes, batch_size, cyclic, tree_name)

        self.seq_maxlen = OrderedDict(sorted(seq_maxlen.iteritems(), key=lambda item: self._example_list.index(item[0])))
        self.padding = padding
        self.truncating = truncating
 
    def _get_data(self, idx): 
        raise NotImplementedError("")
        
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError
            example = self.get_data(key)
            for key, maxlen in self.seq_maxlen.iteritems():
                example[key] = np.expand_dims(example[key], axis=0)
                example[key] = pad_sequences(
                    sequences=example[key],
                    maxlen=maxlen,
                    dtype=np.float32,
                    padding=self.padding,
                    truncating=self.truncating,
                    value=0.)
                example[key] = example[key].reshape(example[key].shape[1:])
            return example
        elif isinstance(key, slice):
            batch = {key: [] for key in self.keys}
            for idx in xrange(*key.indices(len(self))):
                example = self.get_data(idx)
                for key in self.keys:
                    batch[key].append(example[key])
            for key, maxlen in self.seq_maxlen.iteritems():
                batch[key] = pad_sequences(
                    sequences=batch[key],
                    maxlen=maxlen,
                    dtype=np.float32,
                    padding=self.padding,
                    truncating=self.truncating,
                    value=0.)
            batch = {key: np.array(value) for key, value in batch.items()}
            batch = Batch(batch, features=self._features_list, extra=self._extra)
            return batch
        else:
            raise TypeError
