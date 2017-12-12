from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path

import argparse
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')

import keras
from keras.models import load_model
from keras.utils import multi_gpu_model 

from models.layer import add_an_sigmoid_layer
from pipeline import DataLoader

sys.path.append("..")
from meters import ROCMeter, OutHist
from utils import (
    get_log_dir,
    get_saved_model_paths,
    Logger)


def evaluate(saved_model_path,
             step,
             log_dir):

    logger = Logger(log_dir.path, "READ")

    model = load_model(filepath=saved_model_path)

    out_hist = OutHist(
        dpath=log_dir.output_histogram.path,
        step=step,
        dname_list=["train", "test_dijet", "test_zjet"])

    # on training data
    train_data_loader = DataLoader(
        path=logger["train_data"],
        maxlen=logger["maxlen"],
        batch_size=1000,
        cyclic=False)

    for x_daus, x_glob, y in train_data_loader:
        preds = model.predict_on_batch([x_daus, x_glob])
        out_hist.fill(dname="train", labels=y, preds=preds)

    # Test on dijet dataset
    test_dijet_loader = DataLoader(
        path=logger["val_dijet_data"],
        maxlen=logger["maxlen"],
        batch_size=1000,
        cyclic=False)

    roc_dijet = ROCMeter(
        dpath=log_dir.roc.path,
        step=step,
        title="Test on Dijet",
        prefix="dijet_")

    for x_daus, x_glob, y in test_dijet_loader:
        preds = model.predict_on_batch([x_daus, x_glob])

        roc_dijet.append(labels=y, preds=preds)
        out_hist.fill(dname="test_dijet", labels=y, preds=preds)

    roc_dijet.finish()

    # Test on Z+jet dataset
    test_zjet_loader = DataLoader(
        path=logger["val_zjet_data"],
        maxlen=logger["maxlen"],
        batch_size=1000,
        cyclic=False)

    roc_zjet = ROCMeter(
        dpath=log_dir.roc.path,
        step=step,
        title="Test on Z+jet",
        prefix="zjet_")

    for x_daus, x_glob, y in test_zjet_loader:
        preds = model.predict_on_batch([x_daus, x_glob])
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
