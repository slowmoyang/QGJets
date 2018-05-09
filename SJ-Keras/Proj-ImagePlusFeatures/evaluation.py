from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')

import keras
from keras.models import load_model
from keras.utils import multi_gpu_model 
import tensorflow as tf


sys.path.append("..")
from keras4jet.models import get_custom_objects
from keras4jet.data_loader import HybridIFLoader
from keras4jet.meters import OutHist
from keras4jet.meters import ROCMeter
from keras4jet.heatmap import Heatmap
from keras4jet.utils import get_log_dir 
from keras4jet.utils import Config
from keras4jet.utils import get_available_gpus
from keras4jet.utils import get_saved_model_paths


def evaluate(saved_model_path,
             step,
             log_dir):
    # TEST
    config = Config(log_dir.path, "READ")

    custom_objects = get_custom_objects(config.model_type, config.model)

    model = load_model(saved_model_path, custom_objects=custom_objects)

    #model = multi_gpu_model(model, 2)

    out_hist = OutHist(
        dpath=log_dir.output_histogram.path,
        step=step,
        dname_list=["train", "test_dijet", "test_zjet"])


    ##########################
    # training data
    ###########################
    train_loader = HybridIFLoader(
        path=config.training_set,
        features=config.features,
        image=config.image,
        image_shape=config.image_shape,
        batch_size=1024,
        cyclic=False)

    for batch in train_loader:
        y_pred = model.predict_on_batch([batch["image"], batch["features"]])
        out_hist.fill(dname="train", y_true=batch["y"], y_pred=y_pred)


    #############################
    # Test on dijet dataset
    ########################
    dijet_loader = HybridIFLoader(
        path=config.dijet_test_set,
        features=config.features,
        image=config.image,
        image_shape=config.image_shape,
        extra=["pt", "eta"],
        batch_size=1024,
        cyclic=False)

    roc_dijet = ROCMeter(
        dpath=log_dir.roc.path,
        step=step,
        title="Test on Dijet",
        prefix="dijet_")

    heatmap_subdir = os.path.join(
        log_dir.heatmap.path,
        os.path.basename(saved_model_path).split(".")[0])

    if not os.path.exists(heatmap_subdir):
        os.mkdir(heatmap_subdir)

    heatmap = Heatmap(
        data_set=config.dijet_test_set,
        out_dir=heatmap_subdir)

    for batch in dijet_loader:
        y_pred = model.predict_on_batch([batch["image"], batch["features"]])
        roc_dijet.append(y_true=batch["y"], y_pred=y_pred)
        out_hist.fill(dname="test_dijet", y_true=batch["y"], y_pred=y_pred)
        heatmap.fill(y_true=batch["y"], y_pred=y_pred, pt=batch["pt"], eta=batch["eta"])

    roc_dijet.finish()
    heatmap.finish()

    ##################################
    # Test on Z+jet dataset
    ###################################
    test_zjet_loader = HybridIFLoader(
        path=config.zjet_test_set,
        features=config.features,
        image=config.image,
        image_shape=config.image_shape,
        batch_size=1024,
        cyclic=False)

    roc_zjet = ROCMeter(
        dpath=log_dir.roc.path,
        step=step,
        title="Test on Z+jet",
        prefix="zjet_"
    )

    for batch in test_zjet_loader:
        y_pred = model.predict_on_batch([batch["image"], batch["features"]])
        roc_zjet.append(y_true=batch["y"], y_pred=y_pred)
        out_hist.fill("test_zjet", y_true=batch["y"], y_pred=y_pred)

    roc_zjet.finish()

    out_hist.finish()


def evaluate_all(log_dir):
    if isinstance(log_dir, str):
        log_dir = get_log_dir(log_dir, creation=False)

    path_and_step = get_saved_model_paths(log_dir.saved_models.path)
    for i, (saved_model_path, step) in enumerate(path_and_step):
        print("\n\n\n[{i}/{total}]: {path}".format(
            i=i, total=len(path_and_step), path=saved_model_path))
        evaluate(
            saved_model_path,
            step,
            log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir', type=str, required=True,
    	help='the directory path of dataset')
    args = parser.parse_args()
    log_dir = get_log_dir(path=args.log_dir, creation=False)
    evaluate_all(log_dir)
