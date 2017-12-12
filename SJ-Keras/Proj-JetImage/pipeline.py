from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
import numpy as np


class DataLoader(object):

    def __init__(self, path, batch_size, cyclic,
                 image_shape=(3, 33, 33),
                 namecycle="jetAnalyser"):
        self.path = path
        self.batch_size = batch_size
        self.cyclic = cyclic

        self.image_shape = image_shape
        self.namecycle = namecycle
        
        self.root_file = ROOT.TFile(path, "READ")
        self.tree = self.root_file.Get(namecycle)
        
        self._start = 0
        
    def __len__(self):
        return int(self.tree.GetEntries())
    
    def _get_data(self, idx):
        self.tree.GetEntry(idx)
        image = np.array(
            [self.tree.image_cpt33,
             self.tree.image_npt33,
             self.tree.image_cmult33],
            dtype=np.float32)
        image = np.reshape(image, self.image_shape)

        label = np.int64(self.tree.label[1])
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
       


class OldDataLoader(object):
    def __init__(self, path, batch_size, cyclic, image_shape=(3, 33, 33)):
        self.path = path
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.image_shape = image_shape
        
        self.root_file = ROOT.TFile(path, "READ")
        self.tree = self.root_file.Get("jet")
        if self.tree == ROOT.MakeNullPointer():
            print("ERROR:", self.root_file, "contains no 'jet' tree")
            exit(1)
        
        self._start = 0
        
    def __len__(self):
        return int(self.tree.GetEntries())
    
    def _get_data(self, idx):
        self.tree.GetEntry(idx)
        image = np.array(self.tree.image, dtype=np.float32)
        image = np.reshape(image, self.image_shape)
        label = np.int64(self.tree.label[1])
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
       

