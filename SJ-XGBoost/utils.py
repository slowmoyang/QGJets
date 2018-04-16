from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import json
import argparse
import numpy as np
import warnings

class Directory(object):
    def __init__(self, path, creation=True):
        self.path = path
        self._creation = creation
        if self._creation:
            os.makedirs(self.path)

    def make_subdir(self, name):
        path = os.path.join(self.path, name)
        setattr(self, name, Directory(path, creation=self._creation))

class Config(object):
    def __init__(self, dpath, mode="write"):
        self.path = os.path.join(dpath, "config.json")
        self._mode = mode.lower()
        if self._mode == "write":
            self.log = {}
        elif self._mode == "read":
            self.log = self.load(self.path)
            for key, value in self.log.iteritems():
                setattr(self, key, value)

    def update(self, data):
        if isinstance(data, argparse.Namespace):
            data = vars(data)
        self.log.update(data)
        for key, value in data.iteritems():
            setattr(self, key, value)

    def __setitem__(self, key, item):
        self.log[key] = item
        setattr(self, key, item)

    def __getitem__(self, key):
        return self.log[key]

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.log, f, indent=4, sort_keys=True)

    def load(self, path):
        log = open(path).read()
        log = json.loads(log)
        log = dict(log)
        return log

    def finish(self):
        self.save() 





def get_path_of_datasets(event_type, min_pt, max_pt=None):
    if max_pt is None:
        max_pt = min_pt + 100
    path_fmt = "./Data/old_dset/pt_{}_{}/{}_{{}}_set.npz".format(min_pt, max_pt, event_type)

    paths = [path_fmt.format(each) for each in ["training", "validation", "test"]]

    for each in paths:
        if not os.path.exists(each):
            raise IOError("{} doesn't exit!".format(each))

    return paths
    
