from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from datetime import datetime
import shutil

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import sklearn as skl

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import multi_gpu_model 
from tensorflow.keras.utils import plot_model

import keras4hep as kh
from keras4hep.data import get_class_weight
from keras4hep.metrics import roc_auc
from keras4hep.utils.misc import get_available_gpus
from keras4hep.utils.misc import Directory
from keras4hep.utils.misc import Config
from keras4hep.utils.misc import find_good_checkpoint
from keras4hep.projects.qgjets.utils import get_dataset_paths
from keras4hep.projects.toptagging import ROCCurve 
from keras4hep.projects.toptagging import BinaryClassifierResponse
from keras4hep.projects.toptagging import LearningCurve
from keras4hep.utils.misc import parse_str

from dataset import get_data_iter
from model import build_model


def evaluate(checkpoint_path,
             train_iter,
             test_iter,
             log_dir):

    model = load_model(checkpoint_path)

    ckpt_name = os.path.basename(checkpoint_path).replace(".hdf5", "")
    epoch = parse_str(ckpt_name, target="epoch")
    name = "model_epoch-{:02d}".format(epoch)

    title = "Quark/Gluon Jet Discrimination (Epoch {})".format(epoch)

    roc_curve = ROCCurve(
        name=name,
        title=title,
        directory=log_dir.roc_curve.path)

    model_response = BinaryClassifierResponse(
        name=name,
        title=title,
        directory=log_dir.model_response.path)

    ##########################
    # training data
    ###########################
    print("TRAINING SET")
    for batch_idx, batch in enumerate(train_iter, 1):

        y_score = model.predict_on_batch([batch.x_kin, batch.x_pid, batch.x_len])
        model_response.append(is_train=True,
                              y_true=batch.y,
                              y_score=y_score)

    #############################
    # Test on dijet dataset
    ########################
    print("TEST SET")
    for batch_idx, batch in enumerate(test_iter, 1):

        y_score = model.predict_on_batch([batch.x_kin, batch.x_pid, batch.x_len])
        model_response.append(is_train=False,
                              y_true=batch.y,
                              y_score=y_score)
        roc_curve.append(y_true=batch.y, y_score=y_score)

    roc_curve.finish()
    model_response.finish()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="rnn{}".format(datetime.now().strftime("%y%m%d%H%M%S")))
    parser.add_argument("--directory", default="./logs")

    # GPU
    parser.add_argument("--num_gpus", default=len(get_available_gpus()), type=int)
    parser.add_argument("--multi-gpu", default=False, action='store_true', dest='multi_gpu')

    # Hyperparameters
    parser.add_argument("--epoch", dest="epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--valid_batch_size", default=1024, type=int)

    # Optimizer
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--clipnorm", default=-1, type=float,
                        help="if it is greater than 0, then graidient clipping is activated")
    parser.add_argument("--clipvalue", default=-1, type=float)
    parser.add_argument("--use-class-weight", dest="use_class_weight",
                        default=False, action="store_true")

    # Frequencies
    parser.add_argument("--valid_freq", type=int, default=32)
    parser.add_argument("--save_freq", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")

    # Project parameters
    parser.add_argument("--min-pt", dest="min_pt", default=100, type=int)

    # Model Archtecture
    parser.add_argument("--act", dest="activation", default="elu", type=str)
    parser.add_argument("--rnn", default="gru", type=str)

    args = parser.parse_args()

    ###################
    #
    ###################
    log_dir = Directory(path=os.path.join(args.directory, args.name))
    log_dir.mkdir("script")
    log_dir.mkdir("checkpoint")
    log_dir.mkdir("learning_curve")
    log_dir.mkdir("roc_curve")
    log_dir.mkdir("model_response")

    config = Config(log_dir.path, "w")
    config.append(args)
    config["hostname"] = os.environ["HOSTNAME"]
    config["log_dir"] = log_dir.path
    config.save()

    scripts = [
        "./dataset.py",
        "./model.py",
        "./train.py",
    ]
    for each in scripts:
        shutil.copy2(each, log_dir.script.path)
    open(log_dir.script.concat("__init__.py"), 'w').close()
    

    ########################################
    # Load training and validation datasets
    ########################################
    dset = get_dataset_paths(config.min_pt)
    config.append(dset)

    config["seq_maxlen"] = {
        "x_kin": 30,
        "x_pid": 30
    }

    train_iter = get_data_iter(
        path=dset["training"],
        batch_size=config.batch_size,
        seq_maxlen=config.seq_maxlen,
        fit_generator_mode=True)

    valid_iter = get_data_iter(
        path=dset["validation"],
        batch_size=config.valid_batch_size,
        seq_maxlen=config.seq_maxlen,
        fit_generator_mode=True)

    test_iter = get_data_iter(
        path=dset["test"],
        batch_size=config.valid_batch_size,
        seq_maxlen=config.seq_maxlen,
        fit_generator_mode=False)

    if config.use_class_weight: 
        class_weight = get_class_weight(train_iter)
        config["class_weight"] = list(class_weight)
    else:
        class_weight = None


    #################################
    # Build & Compile a model.
    #################################
    x_kin_shape = train_iter.get_shape("x_kin", batch_shape=False)
    x_pid_shape = train_iter.get_shape("x_pid", batch_shape=False)

    model = build_model(
        x_kin_shape,
        x_pid_shape,
        rnn=config.rnn,
        activation=config.activation,
        name=config.name)

    config["model"] = model.get_config()


    if config.multi_gpu:
        model = multi_gpu_model(model, gpus=config.num_gpus)

    if config.hostname == "cms05.sscc.uos.ac.kr":
        model_plot_path = log_dir.concat("model.png")
        plot_model(model, to_file=model_plot_path, show_shapes=True)
    else:
        model.summary()

    loss = 'categorical_crossentropy'

    # TODO capsulisation
    optimizer_kwargs = {}
    if config.clipnorm > 0:
        optimizer_kwargs["clipnorm"] = config.clipnorm
    if config.clipvalue > 0:
        optimizer_kwargs["clipvalue"] = config.clipvalue
    optimizer = getattr(optimizers, config.optimizer)(lr=config.lr, **optimizer_kwargs)

    metric_list = ["accuracy" , roc_auc]

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metric_list)

    config["loss"] = loss
    config["optimizer_config"] = optimizer.get_config()

    ###########################################################################
    # Callbacks
    ###########################################################################
    ckpt_format_str = "weights_epoch-{epoch:02d}_loss-{val_loss:.4f}_acc-{val_acc:.4f}_auc-{val_roc_auc:.4f}.hdf5"
    ckpt_path = log_dir.checkpoint.concat(ckpt_format_str)
    csv_log_path = log_dir.concat("log_file.csv")

    learning_curve = LearningCurve(directory=log_dir.learning_curve.path)
    learning_curve.book(x="step", y="roc_auc", best="max")
    learning_curve.book(x="step", y="acc", best="max")
    learning_curve.book(x="step", y="loss", best="min")

    callback_list = [
        callbacks.ModelCheckpoint(filepath=ckpt_path),
        callbacks.ReduceLROnPlateau(verbose=1),
        callbacks.CSVLogger(csv_log_path),
        learning_curve,
    ]

    ############################################################################
    # Training
    ############################################################################
    model.fit_generator(
        train_iter,
        steps_per_epoch=len(train_iter),
        epochs=config.epochs,
        validation_data=valid_iter,
        validation_steps=len(valid_iter),
        callbacks=callback_list,
        shuffle=True,
        class_weight=class_weight)

    del model

    print("Training is over! :D")


    ###########################################
    # Evaluation
    ############################################
    train_iter.fit_generator_mode = False
    train_iter.cycle = False

    good_ckpt = find_good_checkpoint(
        log_dir.checkpoint.path,
        which={"max": ["auc", "acc"], "min": ["loss"]})

    all_ckpt = set(log_dir.checkpoint.get_entries())
    # no local optima
    useless_ckpt = all_ckpt.difference(good_ckpt)
    for each in useless_ckpt:
        os.remove(each)
    
    for idx, each in enumerate(good_ckpt, 1):
        print("[{}/{}] {}".format(idx, len(good_ckpt), each))

        K.clear_session()
        evaluate(checkpoint_path=each, 
                 train_iter=train_iter,
                 test_iter=test_iter,
                 log_dir=log_dir)

    config.save()


if __name__ == "__main__":
    from ROOT import gROOT
    gROOT.SetBatch(True)
    main()
