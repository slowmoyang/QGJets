from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import json
import argparse
import keras.backend as K
import numpy as np
import warnings

from tensorflow.python.client import device_lib

class Directory(object):
    def __init__(self, path, creation=True):
        self.path = path
        self._creation = creation
        if self._creation:
            os.makedirs(self.path)

    def make_subdir(self, name):
        path = os.path.join(self.path, name)
        setattr(self, name, Directory(path, creation=self._creation))

    def get_entries(self, full_path=True):
        entries = os.listdir(self.path)
        if full_path:
            entries = map(lambda each: os.path.join(self.path, each), entries)
        return entries



def get_log_dir(path, creation=True):
    # mkdir
    log = Directory(path, creation=creation)
    log.make_subdir('validation')
    log.make_subdir('saved_models')
    log.make_subdir('roc')
    log.make_subdir('output_histogram')
    log.make_subdir("heatmap")
    return log


def get_saved_model_paths(dpath):
    def foo(f):
        step = f.split("_")[1].split(".")[0]
        path = os.path.join(dpath, f)
        return (path, step)

    saved_models = os.listdir(dpath)
    saved_models.sort()
    saved_models = map(foo, saved_models)
    return saved_models


def write_args_to_json_file(args, log_dir):
    args_dict = vars(args)

    path = os.path.join(log_dir, "args.json")
    with open(path, "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=True)


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


class Logger(object):
    def __init__(self, dpath, mode="write"):
        warnings.warn(
            "In the future The name of the logger will be changed to config.",
            FutureWarning)

        self.path = os.path.join(dpath, "args.json")
        self._mode = mode.lower()
        if self._mode == "write":
            self.log = {}
        elif self._mode == "read":
            self.log = self.load(self.path)

    def update(self, data):
        if isinstance(data, dict):
            self.log.update(data)
        elif isinstance(data, argparse.Namespace):
            self.log.update(vars(data))

    def __setitem__(self, key, item):
        self.log[key] = item

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

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_dataset_paths(dpath):
    entries = os.listdir(dpath)
    datasets = {}
    for each in entries:
        key, _ = os.path.splitext(each)
        datasets[key] = os.path.join(dpath, each)

    return datasets


def get_size_of_model(model):
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])


def get_filename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def convert_str_to_number(string):
    float_case = float(string)
    int_case = int(float_case)
    output = int_case if float_case == int_case else float_case
    return output

