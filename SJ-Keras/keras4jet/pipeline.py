from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
import numpy as np

from numba import jit

class OldImageDataLoader(object):

    def __init__(self, path, x, x_shape,
                 batch_size, cyclic,
                 namecycle="jetAnalyser"):
        self.path = path
        self.batch_size = batch_size
        self.cyclic = cyclic

        self.x = x
        self.x_shape = x_shape
        self.namecycle = namecycle
        
        self.root_file = ROOT.TFile(path, "READ")
        self.tree = self.root_file.Get(namecycle)
        
        self._start = 0
        
    def __len__(self):
        return int(self.tree.GetEntries())
    
    def _get_data(self, idx):
        self.tree.GetEntry(idx)
        image = np.array(
            object=[getattr(self.tree, each) for each in self.x],
            dtype=np.float32).reshape(self.x_shape)
        label = np.array(self.tree.label, dtype=np.int64)
        return (image, label)
        
    
    @jit
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError
            return self._get_data(key)
        elif isinstance(key, slice):
            x = []
            y = []
            for idx in xrange(*key.indices(len(self))):
                image, label = self._get_data(idx)
                x.append(image)
                y.append(label)
            x = np.array(x)
            y = np.array(y)
            return (x, y)
        else:
            raise TypeError
    
    @jit    
    def next(self):
        if self.cyclic:
            if self._start + 1 < len(self):
                end = self._start + self.batch_size
                slicing = slice(self._start, end)
                if end <= len(self):
                    self._start = end
                    return self[slicing]
                else:
                    x, y = self[slicing]
                    
                    self._start = 0
                    end = end - len(self)

                    x1, y1 = self[slice(self._start, end)]
                    self._start = end
                    
                    np.append(x, x1, axis=0)
                    np.append(y, y1, axis=0)
                    return x, y
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
       


class ImageDataLoader(object):

    def __init__(self,
                 path,
                 x,
                 x_shape,
                 batch_size,
                 cyclic,
                 y="label",
                 num_classes=2,
                 namecycle="jetAnalyser"):
        self.path = path
        self.batch_size = batch_size
        self.cyclic = cyclic

        self.x = x
        self.x_shape = x_shape
        self.y = y
        self.num_classes = num_classes
        self.namecycle = namecycle
        
        self.root_file = ROOT.TFile.Open(path, "READ")
        self.tree = self.root_file.Get(namecycle)
        
        self._start = 0
        
    def __len__(self):
        return int(self.tree.GetEntries())
    
    def _get_data(self, idx):
        self.tree.GetEntry(idx)
        image = np.array(
            object=[getattr(self.tree, each) for each in self.x],
            dtype=np.float32).reshape(self.x_shape)

        label = np.zeros(self.num_classes, dtype=np.int64)
        label[int(getattr(self.tree, self.y))] = 1
        return (image, label)
        
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError
            return self._get_data(key)
        elif isinstance(key, slice):
            x = []
            y = []
            for idx in xrange(*key.indices(len(self))):
                image, label = self._get_data(idx)
                x.append(image)
                y.append(label)
            x = np.array(x)
            y = np.array(y)
            return (x, y)
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
                    x, y = self[slicing]
                    
                    self._start = 0
                    end = end - len(self)

                    x1, y1 = self[slice(self._start, end)]
                    self._start = end
                    
                    np.append(x, x1, axis=0)
                    np.append(y, y1, axis=0)
                    return x, y
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
 


def old_generate_arrays_from_root_file(path, x, x_shape, batch_size, extra=[], y="label"):
    f = ROOT.TFile.Open(path, "READ")
    tree = f.Get("jetAnalyser")
    num_entries = tree.GetEntries()

    for start in xrange(0, num_entries, batch_size):
        example = {"x": [], "y": []}
        example.update({key: [] for key in extra})

        end = start + batch_size 
        end = end if end <= num_entries else num_entries
        for i in range(start, end):
            tree.GetEntry(i)
            example["x"].append(
                np.array(object=[getattr(tree, each) for each in x],
                         dtype=np.float32).reshape(x_shape))
            example["y"].append(np.array([getattr(tree, y)], dtype=np.int64).reshape(-1))
            for key in extra:
                example[key].append(getattr(tree, key))
        example = {key: np.array(value) for key, value in example.items()}
        yield example


def generate_arrays_from_root_file(path,
                                   x,
                                   x_shape,
                                   batch_size,
                                   extra=[],
                                   y="label",
                                   num_classes=2):
    f = ROOT.TFile.Open(path, "READ")
    tree = f.Get("jetAnalyser")
    num_entries = tree.GetEntries()

    for start in xrange(0, num_entries, batch_size):
        example = {"x": [], "y": []}
        example.update({key: [] for key in extra})

        end = start + batch_size 
        end = end if end <= num_entries else num_entries
        for i in range(start, end):
            tree.GetEntry(i)

            image = np.array(
                object=[getattr(tree, each) for each in x],
                dtype=np.float32)
            image = image.reshape(x_shape)

            label = np.zeros(num_classes, dtype=np.int64)
            label[int(getattr(tree, y))] = 1

            example["x"].append(image)
            example["y"].append(label)
            for key in extra:
                example[key].append(getattr(tree, key))
        example = {key: np.array(value) for key, value in example.items()}
        yield example

