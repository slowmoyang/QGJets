from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
import numpy as np

import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences

from keras.backend.tensorflow_backend import set_session


class SeqDataLoader(object):
    """
      ref. CMS AN-17-188
    """
    def __init__(self, path, batch_size, cyclic, maxlen=54, tree_name="jetAnalyser"):
        """
        """
        self.path = path
        self.batch_size = batch_size
        self.cyclic = cyclic
        
        self.root_file = ROOT.TFile(path, "READ")
        self.tree = self.root_file.Get(tree_name)
        if self.tree == ROOT.MakeNullPointer():
            print("ERROR:", self.root_file, "contains no 'jet' tree")
            exit(1)

        if maxlen is None:
            self.maxlen = int(self.tree.GetMaximum("n_dau"))
        else:
            self.maxlen = maxlen

        self.x_daus_shape = (self.maxlen, 5)
        self.x_glob_shape = (4, )

        self._start = 0
        
    def __len__(self):
        return int(self.tree.GetEntries())
    
    def _get_data(self, idx):
        self.tree.GetEntry(idx)

        # x_dau
        # daughter features used as input
        dau_pt = np.array(self.tree.dau_pt, dtype=np.float32)
        dau_deta = np.array(self.tree.dau_deta, dtype=np.float32)
        dau_dphi = np.array(self.tree.dau_dphi, dtype=np.float32)
        dau_charge = np.array(self.tree.dau_charge, dtype=np.float32)

        pt_ordering = np.argsort(dau_pt)

        x_daus = np.vstack((dau_pt, dau_deta, dau_dphi, dau_charge))
        x_daus = x_daus.T
        x_daus = x_daus[pt_ordering]

        # x_glob
        # global input features
        x_glob = np.array([
            self.tree.pt,
            self.tree.eta,
            self.tree.phi,
            self.tree.n_dau])

        # y
        # label
        y = np.int64(self.tree.label[1])
        return (x_daus, x_glob, y)
        
    
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError
            x_daus, x_glob, y = self._get_data(key)

            # CMS AN-17-188.
            # Sec. 3.1 Slim jet DNN architecture (p. 11 / line 196-198)
            # When using recurrent networks the ordering is important, thus
            # our underlying assumption is that the most displaced (in case
            # of displacement) or the highest pT candidates matter the most.

            x_daus = np.expand_dims(x_daus, axis=0)
            x_daus = pad_sequences(
                sequences=x_daus,
                maxlen=self.maxlen,
                dtype=np.float32,
                padding="pre",
                truncating="pre",
                value=0.)
            x_daus = x_daus.reshape(x_daus.shape[1:])

            return x_daus, x_glob, y

        elif isinstance(key, slice):
            x_daus_batch = []
            x_glob_batch = []
            y_batch = []
            for idx in xrange(*key.indices(len(self))):
                x_daus, x_glob, y = self._get_data(idx)

                x_daus_batch.append(x_daus)
                x_glob_batch.append(x_glob)
                y_batch.append(y)

            x_daus_batch = pad_sequences(
                sequences=x_daus_batch,
                maxlen=self.maxlen,
                dtype=np.float32,
                padding="pre",
                truncating="pre",
                value=0.)

            x_glob_batch = np.array(x_glob_batch)
            y_batch = np.array(y_batch)

            return x_daus_batch, x_glob_batch, y_batch
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
                    x_daus, x_glob, y = self[slicing]
                    
                    self._start = 0
                    end = end - len(self)

                    x1_daus, x1_glob, y1 = self[slice(self._start, end)]
                    self._start = end
                    
                    np.append(x_daus, x1_daus, axis=0)
                    np.append(x_glob, x1_glob, axis=0)
                    np.append(y, y1, axis=0)
                    return x_daus, x_glob, y
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
       
    def get_x_daus_shape(self):
        return self.x_daus_shape

    def get_x_glob_shape(self):
        return self.x_glob_shape



class DataLoader(object):
    """
      ref. CMS AN-17-188
    """
    def __init__(self, path, batch_size, cyclic, maxlen=54, tree_name="jetAnalyser"):
        """
        """
        self.path = path
        self.batch_size = batch_size
        self.cyclic = cyclic
        
        self.root_file = ROOT.TFile(path, "READ")
        self.tree = self.root_file.Get(tree_name)
        if self.tree == ROOT.MakeNullPointer():
            print("ERROR:", self.root_file, "contains no '", tree_name, "' tree")
            exit(1)

        if maxlen is None:
            self.maxlen = int(self.tree.GetMaximum("n_dau"))
        else:
            self.maxlen = maxlen

        self._start = 0
        
    def __len__(self):
        return int(self.tree.GetEntries())
    
    def _get_data(self, idx):
        self.tree.GetEntry(idx)

        # x_dau
        # daughter features used as input
        dau_pt = np.array(self.tree.dau_pt, dtype=np.float32)
        dau_rel_pt = dau_pt / self.tree.pt

        dau_deta = np.array(self.tree.dau_deta, dtype=np.float32)
        dau_dphi = np.array(self.tree.dau_dphi, dtype=np.float32)

        dau_charge = np.array(self.tree.dau_charge, dtype=np.float32)
        #is_charged = np.where(dau_charge != 0, 0, 1)
        is_neutral = np.where(dau_charge == 0, 1, 0)

        pt_ordering = np.argsort(dau_pt)

        x_daus = np.vstack((
            dau_rel_pt,
            dau_deta,
            dau_dphi,
            dau_charge,
            is_neutral
        ))
        
        x_daus = x_daus.T
        x_daus = x_daus[pt_ordering]

        # x_glob
        # global input features
        x_glob = np.array([
            self.tree.pt,
            self.tree.eta,
            self.tree.phi,
            self.tree.n_dau])

        # y
        # label
        y = np.int64(self.tree.label[1])
        return (x_daus, x_glob, y)
        
    
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError
            x_daus, x_glob, y = self._get_data(key)

            # CMS AN-17-188.
            # Sec. 3.1 Slim jet DNN architecture (p. 11 / line 196-198)
            # When using recurrent networks the ordering is important, thus
            # our underlying assumption is that the most displaced (in case
            # of displacement) or the highest pT candidates matter the most.

            x_daus = np.expand_dims(x_daus, axis=0)
            x_daus = pad_sequences(
                sequences=x_daus,
                maxlen=self.maxlen,
                dtype=np.float32,
                padding="pre",
                truncating="pre",
                value=0.)
            x_daus = x_daus.reshape(x_daus.shape[1:])

            return x_daus, x_glob, y

        elif isinstance(key, slice):
            x_daus_batch = []
            x_glob_batch = []
            y_batch = []
            for idx in xrange(*key.indices(len(self))):
                x_daus, x_glob, y = self._get_data(idx)

                x_daus_batch.append(x_daus)
                x_glob_batch.append(x_glob)
                y_batch.append(y)

            x_daus_batch = pad_sequences(
                sequences=x_daus_batch,
                maxlen=self.maxlen,
                dtype=np.float32,
                padding="pre",
                truncating="pre",
                value=0.)

            x_glob_batch = np.array(x_glob_batch)
            y_batch = np.array(y_batch)

            return x_daus_batch, x_glob_batch, y_batch
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
                    x_daus, x_glob, y = self[slicing]
                    
                    self._start = 0
                    end = end - len(self)

                    x1_daus, x1_glob, y1 = self[slice(self._start, end)]
                    self._start = end
                    
                    np.append(x_daus, x1_daus, axis=0)
                    np.append(x_glob, x1_glob, axis=0)
                    np.append(y, y1, axis=0)
                    return x_daus, x_glob, y
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
       
    def get_shapes(self):
        inputs = self[0][:-1]
        inputs_shape = {"input{}_shape".format(i): x.shape for i, x in enumerate(inputs, 0)}
        return inputs_shape

