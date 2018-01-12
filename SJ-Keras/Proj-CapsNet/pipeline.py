from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ROOT
import numpy as np


class DataLoader(object):

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
        label = np.array(self.tree.label, dtype=np.int64).reshape(-1)
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
       

def generate_arrays_from_root_file(path, x, x_shape, batch_size, extra=[], y="label"):
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
                         dtype=np.float32))
            example["y"].append(np.int64([getattr(tree, y)]))
            for key in extra:
                example[key].append(getattr(tree, key))
        example = {key: np.array(value) for key, value in example.items()}
        yield example


if __name__ == "__main__":
    import time
    generator = generate_arrays_from_root_file(
        path="../../Data/pt_100_500/dijet_set/dijet_test_set.root",
        x=["image_cpt_33", "image_nhad_33",
           "image_photon_33", "image_cmult_33"],
        x_shape=(4, 33, 33),
        y="label",
        extra=["pt", "eta"],
        batch_size=500)

    start = time.time()
    for i, example in enumerate(generator):
        continue
    duration = time.time() - start
    print("Total step: {}".format(i))
    print("Duration: {:.2f} sec".format(duration))
    print("Time per batch: {:.4f} sec".format(duration/i))


    print("\n#################################\n")
    start = time.time()
    data_loader = DataLoader(
        path="../../Data/pt_100_500/dijet_set/dijet_test_set.root",
        batch_size=500, cyclic=False)

    for i, (x, y) in enumerate(data_loader):
        continue
    duration = time.time() - start
    print("Total step: {}".format(i))
    print("Duration: {:.2f} sec".format(duration))
    print("Time per batch: {:.4f} sec".format(duration/i))

