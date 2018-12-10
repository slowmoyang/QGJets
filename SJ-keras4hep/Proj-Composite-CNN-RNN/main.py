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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

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

from cnn_script.model import get_custom_objects as get_cnn_custom_objects
from rnn_script.model import get_custom_objects as get_rnn_custom_objects
from dataset import get_data_iter


def backup_scripts(directory):
    sources = [
        "./dataset.py",
        "./main.py",
    ]
    for each in sources:
        shutil.copy2(each, directory)

def evaluate(checkpoint_path,
             train_iter,
             test_iter,
             log_dir,
             custom_objects={}):

    model = load_model(checkpoint_path, custom_objects=custom_objects)

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

        y_score = model.predict_on_batch([batch.x_img, batch.x_kin, batch.x_pid, batch.x_len])
        model_response.append(is_train=True,
                              y_true=batch.y,
                              y_score=y_score)

    #############################
    # Test on dijet dataset
    ########################
    print("TEST SET")
    for batch_idx, batch in enumerate(test_iter, 1):

        y_score = model.predict_on_batch([batch.x_img, batch.x_kin, batch.x_pid, batch.x_len])
        model_response.append(is_train=False,
                              y_true=batch.y,
                              y_score=y_score)
        roc_curve.append(y_true=batch.y, y_score=y_score)

    roc_curve.finish()
    model_response.finish()




def main():
    ##########################
    # Argument Parsing
    ##########################
    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", dest="log_dir", type=str,
                        default="./logs/untitled-{}".format(datetime.now().strftime("%y%m%d-%H%M%S")))

    parser.add_argument("--num_gpus", default=len(get_available_gpus()), type=int)
    parser.add_argument("--multi-gpu", default=False, action='store_true', dest='multi_gpu')

    # Hyperparameters
    parser.add_argument("--epoch", dest="epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--valid_batch_size", default=1024, type=int)

    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--lr", default=0.003, type=float)
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

    args = parser.parse_args()

    ###################
    #
    ###################
    log_dir = Directory(path=args.log_dir)
    log_dir.mkdir("script")
    log_dir.mkdir("checkpoint")
    log_dir.mkdir("learning_curve")
    log_dir.mkdir("roc_curve")
    log_dir.mkdir("model_response")

    backup_scripts(log_dir.script.path)

    config = Config(log_dir.path, "w")
    config.append(args)
    config["hostname"] = os.environ["HOSTNAME"]

    ###############################
    # Load
    #################################
    if os.environ["HOSTNAME"] == "cms05.sscc.uos.ac.kr":
        ckpt_dir = "/store/slowmoyang/QGJets/SJ-keras4hep/Dev-Composite"
    elif os.environ["HOSTNAME"] == "gate2":
        ckpt_dir = "/scratch/slowmoyang/QGJets/SJ-keras4hep/Dev-Composite"
    else:
        raise NotImplementedError

    cnn_path = os.path.join(ckpt_dir, "VanillaConvNet_epoch-67_loss-0.4987_acc-0.7659_auc-0.8422.hdf5")
    rnn_path = os.path.join(ckpt_dir, "RNNGatherEmbedding_weights_epoch-121_loss-0.4963_acc-0.7658_auc-0.8431.hdf5")

    cnn_custom_objects = get_cnn_custom_objects()
    rnn_custom_objects = get_rnn_custom_objects()

    custom_objects = {}
    custom_objects.update(cnn_custom_objects)
    custom_objects.update(rnn_custom_objects)

    cnn = load_model(cnn_path, custom_objects=cnn_custom_objects)
    cnn.summary()
    print("\n"*5)

    rnn = load_model(rnn_path, custom_objects=rnn_custom_objects)
    rnn.summary()
    print("\n"*5)

    ######################################
    # Build
    ######################################
    inputs = cnn.inputs + rnn.inputs

    # cnn_last_hidden = cnn.get_layer("cnn_conv2d_3").output
    # rnn_last_hidden = rnn.get_layer("rnn_dense_5").output
    # cnn_flatten = Flatten()(cnn_last_hidden)
    # joint = Concatenate(axis=-1)([cnn_flatten, rnn_last_hidden])

    cnn_last_hidden = cnn.get_layer("cnn_batch_norm_2").output
    rnn_last_hidden = rnn.get_layer("rnn_dense_5").output

    cnn_gap = GlobalAveragePooling2D()(cnn_last_hidden)
    cnn_flatten = Flatten()(cnn_gap)
    joint = Concatenate(axis=-1)([cnn_flatten, rnn_last_hidden])
    joint = BatchNormalization(axis=-1, name="joint_batch_norm")(joint)
    joint = Dense(128)(joint)
    joint = Activation("relu")(joint)
    logits = Dense(2)(joint)

    y_pred = Softmax()(logits)

    model = Model(inputs=inputs, outputs=y_pred)

    model.summary()

    ################################################
    # Freeze
    ##################################################
    for each in model.layers:
        if each.name.startswith("cnn") or each.name.startswith("rnn"):
            each.trainable = False

    ###################################################
    #
    ####################################################
    dset = get_dataset_paths(config.min_pt)
    config["fit_generator_input"] = {
        "x": ["x_img", "x_kin", "x_pid", "x_len"],
        "y": ["y"]
    }

    train_iter = get_data_iter(
        path=dset["training"],
        batch_size=config.batch_size,
        fit_generator_input=config.fit_generator_input,
        fit_generator_mode=True)

    valid_iter = get_data_iter(
        path=dset["validation"],
        batch_size=config.valid_batch_size,
        fit_generator_input=config.fit_generator_input,
        fit_generator_mode=True)

    test_iter = get_data_iter(
        path=dset["test"],
        batch_size=config.valid_batch_size,
        fit_generator_input=config.fit_generator_input,
        fit_generator_mode=False)

    if config.use_class_weight: 
        class_weight = get_class_weight(train_iter)
        config["class_weight"] = list(class_weight)
    else:
        class_weight = None



    ######################################
    #
    #######################################
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
#         callbacks.EarlyStopping(monitor="val_loss" , patience=5),
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

    print("Training is over! :D")

    del model

    ###########################################
    # Evaluation
    ############################################
    train_iter.fit_generator_mode = False
    train_iter.cycle = False

    good_ckpt = find_good_checkpoint(
        log_dir.checkpoint.path,
        which={"max": ["auc", "acc"], "min": ["loss"]})

    for idx, each in enumerate(good_ckpt, 1):
        print("[{}/{}] {}".format(idx, len(good_ckpt), each))

        K.clear_session()
        evaluate(custom_objects=custom_objects,
                 checkpoint_path=each, 
                 train_iter=train_iter,
                 test_iter=test_iter,
                 log_dir=log_dir)

    config.save()


if __name__ == "__main__":
    main()
