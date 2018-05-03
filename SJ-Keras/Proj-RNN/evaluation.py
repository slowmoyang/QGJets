from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
from keras4jet.data_loader import AK4Loader
from keras4jet.metrics_wrapper import roc_auc_score
from keras4jet.meters import OutHist
from keras4jet.meters import ROCMeter
from keras4jet.heatmap import Heatmap
from keras4jet.utils import get_log_dir 
from keras4jet.utils import Config
from keras4jet.utils import get_available_gpus
from keras4jet.utils import get_saved_model_paths
from keras4jet.utils import get_filename
from keras4jet.eval_utils import find_good_models
from keras4jet.eval_utils import parse_model_path


def evaluate(saved_model_path,
             step,
             log_dir):
    # TEST
    config = Config(log_dir.path, "READ")

    custom_objects = get_custom_objects(config.model_type, config.model)
    custom_objects.update({"roc_auc_score": roc_auc_score})

    model = load_model(saved_model_path, custom_objects=custom_objects)

    #model = multi_gpu_model(model, 2)

    out_hist = OutHist(
        dpath=log_dir.output_histogram.path,
        step=step,
        dname_list=["train", "test_dijet", "test_zjet"])

    ##########################
    # training data
    ###########################
    train_loader = AK4Loader(
        path=config.training_set,
        maxlen=config.maxlen,
        batch_size=1024,
        cyclic=False)

    for batch in train_loader:
        y_pred = model.predict_on_batch([batch["x_daus"], batch["x_glob"]])
        out_hist.fill(dname="train", y_true=batch["y"], y_pred=y_pred)


    #############################
    # Test on dijet dataset
    ########################
    dijet_loader = AK4Loader(
        path=config.dijet_test_set,
        maxlen=config.maxlen,
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
        get_filename(saved_model_path))

    if not os.path.exists(heatmap_subdir):
        os.mkdir(heatmap_subdir)

    heatmap = Heatmap(
        data_set=config.dijet_test_set,
        out_dir=heatmap_subdir)

    for batch in dijet_loader:
        y_pred = model.predict_on_batch([batch["x_daus"], batch["x_glob"]])
        roc_dijet.append(y_true=batch["y"], y_pred=y_pred)
        out_hist.fill(dname="test_dijet", y_true=batch["y"], y_pred=y_pred)
        heatmap.fill(y_true=batch["y"], y_pred=y_pred, pt=batch["pt"], eta=batch["eta"])

    roc_dijet.finish()
    heatmap.finish()

    ##################################
    # Test on Z+jet dataset
    ###################################
    test_zjet_loader = AK4Loader(
        path=config.zjet_test_set,
        maxlen=config.maxlen,
        batch_size=1024,
        cyclic=False)

    roc_zjet = ROCMeter(
        dpath=log_dir.roc.path,
        step=step,
        title="Test on Z+jet",
        prefix="zjet_")

    for batch in test_zjet_loader:
        y_pred = model.predict_on_batch([batch["x_daus"], batch["x_glob"]])
        roc_zjet.append(y_true=batch["y"], y_pred=y_pred)
        out_hist.fill("test_zjet", y_true=batch["y"], y_pred=y_pred)

    roc_zjet.finish()

    out_hist.finish()


def evaluate_all(log_dir):
    if isinstance(log_dir, str):
        log_dir = get_log_dir(log_dir, creation=False)

    good_models = find_good_models(log_dir)
    for i, path in enumerate(good_models):
        step = parse_model_path(path)["step"]
        evaluate(path, step, log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_dir', type=str, required=True,
    	help='the directory path of dataset')
    args = parser.parse_args()
    log_dir = get_log_dir(path=args.log_dir, creation=False)


    evaluate_all(log_dir)
