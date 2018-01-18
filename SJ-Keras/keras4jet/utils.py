from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import json
import argparse

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


class Logger(object):
    def __init__(self, dpath, mode="write"):
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


def get_dataset_paths(dpath, data="dijet"):
    data = data.lower()
    if data in ["dijet", "dijet_set", "dj"]:
        dpath = os.path.join(dpath, "dijet_set")
    elif data in ["zjet", "zjet_set", "zj", "z+jet", "z+jet_set"]:
        dpath = os.path.join(dpath, "zjet_set")
    else:
        pass

    entries = os.listdir(dpath)
    datasets = {}
    for each in entries:
        key = os.path.splitext(each)[0]
        datasets[key] = os.path.join(dpath, each)

    for key in datasets:
        if "train" in key:
            datasets["training_set"] = datasets.pop(key)
            break

    return datasets
