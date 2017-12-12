from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import argparse
from datetime import datetime
from tqdm import tqdm
#import numpy as np
import matplotlib as mpl
mpl.use('Agg')

#import tensorflow as tf

import keras
#from keras import (
#    optimizers,
#    losses,
#    metrics)
from keras.models import load_model
from keras.utils import multi_gpu_model 

#import keras.backend as K

from models import add_an_sigmoid_layer
from pipeline import DataLodaer

from meters import ROCMeter, OutHist
from utils import (
    get_log_dir,
    get_saved_model_paths
)



def evaluate(saved_model_path,
             step,
             log_dir):
    # TEST
    logger = Logger(log_dir.path, "READ")

    model = load_model(saved_model_path)

    model = multi_gpu_model(model, 2)

    out_hist = OutHist(
        dpath=log_dir.output_histogram.path,
        step=step,
        dname_list=["train", "test_dijet", "test_zjet"])

    # on training data
    train_data_loader = DataLodaer(
        path=logger["train_data"],
        batch_size=1000,
        cyclic=False)

    for x, y in train_data_loader:
        preds = model.predict_on_batch(x)
        out_hist.fill(dname="train", labels=y, preds=preds)

    # Test on dijet dataset
    test_dijet_loader = DataLodaer(
        path=logger["val_dijet_data"],
        batch_size=1000,
        cyclic=False)

    roc_dijet = ROCMeter(
        dpath=log_dir.roc.path,
        step=step,
        title="Test on Dijet",
        prefix="dijet_"
    )

    for x, y in test_dijet_loader:
        preds = model.predict_on_batch(x)

        roc_dijet.append(labels=y, preds=preds)
        out_hist.fill(dname="test_dijet", labels=y, preds=preds)

    roc_dijet.finish()

    # Test on Z+jet dataset
    test_zjet_loader = DataLodaer(
        path=logger["val_zjet_data"],
        batch_size=1000,
        cyclic=False)

    roc_zjet = ROCMeter(
        dpath=log_dir.roc.path,
        step=step,
        title="Test on Z+jet",
        prefix="zjet_"
    )

    for x, y in test_zjet_loader:
        preds = model.predict_on_batch(x)
        roc_zjet.append(labels=y, preds=preds)
        out_hist.fill("test_zjet", labels=y, preds=preds)

    roc_zjet.finish()

    out_hist.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
	    '--log_dir', type=str, required=True,
    	help='the directory path of dataset')

    args = parser.parse_args()

    log_dir = get_log_dir(path=args.log_dir, creation=False)

    path_and_step = get_saved_model_paths(log_dir.saved_models.path)
    for i, (saved_model_path, step) in enumerate(path_and_step):
        print("\n\n\n[{i}/{total}]: {path}".format(
            i=i, total=len(path_and_step), path=saved_model_path))
        evaluate(
            saved_model_path,
            step,
            log_dir)


if __name__ == "__main__":
    main()
