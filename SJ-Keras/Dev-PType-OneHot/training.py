from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import sys
import os

import matplotlib as mpl
mpl.use("Agg")

from dataset import get_data_iter
from model import build_a_model

sys.path.append("..")
from keras4jet.meters import Meter
from keras4jet import train_utils
from keras4jet.utils import get_log_dir
from keras4jet.utils import Config
from keras4jet.utils import get_available_gpus
from keras4jet.utils import get_dataset_paths

import keras
import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils import multi_gpu_model 

import argparse
import numpy as np
from datetime import datetime

import tensorflow as tf


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets_dir",
                        default="/store/slowmoyang/QGJets/data/root_100_200/2-Refined/",
                        type=str)

    parser.add_argument("--log_dir", default="./logs/{name}", type=str)
    parser.add_argument("--num_gpus", default=len(get_available_gpus()), type=int)
    parser.add_argument("--multi-gpu", default=False, action='store_true', dest='multi_gpu')

    # Hyperparameters
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--valid_batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    # Frequencies
    parser.add_argument("--valid_freq", type=int, default=32)
    parser.add_argument("--save_freq", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")

    # Project parameters

    args = parser.parse_args()

    #########################################################
    # Log directory
    #######################################################
    if '{name}' in args.log_dir:
        args.log_dir = args.log_dir.format(
            name="Untitled_{}".format(
                datetime.today().strftime("%Y-%m-%d_%H-%M-%S")))
    log_dir = get_log_dir(path=args.log_dir, creation=True)
 
    # Config
    config = Config(dpath=log_dir.path, mode="WRITE")
    config.update(args)

    dataset_paths = get_dataset_paths(config.datasets_dir)
    for key, value in dataset_paths.iteritems():
        print("{}: {}".format(key, value))
    config.update(dataset_paths)

    ########################################
    # Load training and validation datasets
    ########################################
    config["seq_maxlen"] = {"x": 40}

    train_iter = get_data_iter(
        path=config.dijet_training_set,
        seq_maxlen=config.seq_maxlen,
        batch_size=config.batch_size)

    valid_dijet_iter = get_data_iter(
        path=config.dijet_validation_set,
        seq_maxlen=config.seq_maxlen,
        batch_size=config.valid_batch_size,
        cyclic=True)

    valid_zjet_iter = get_data_iter(
        path=config.zjet_validation_set,
        seq_maxlen=config.seq_maxlen,
        batch_size=config.valid_batch_size,
        cyclic=True)

    steps_per_epoch = len(train_iter)
    total_step = config.num_epochs * steps_per_epoch
    if config.verbose:
        print("# of steps per one epoch: {}".format(steps_per_epoch))
        print("Total step: {}".format(total_step))


    #################################
    # Build & Compile a model.
    #################################
    x_shape = train_iter.get_shape("x", batch_shape=False)

    _model = build_a_model(x_shape)

    if config.multi_gpu:
        model = multi_gpu_model(_model, gpus=config.num_gpus)
    else:
        model = _model

    # TODO config should have these information.
    loss = 'binary_crossentropy'
    optimizer = optimizers.Adam(lr=config.lr)
    metric_list = ['accuracy']

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metric_list)

    if config.verbose:
        model.summary()

    #######################################
    # 
    ###########################################

    meter = Meter(
        name_list=["step", "lr",
                   "train_loss", "dijet_loss", "zjet_loss",
                   "train_acc", "dijet_acc", "zjet_acc"],
        dpath=log_dir.validation.path)

    #######################################
    # Training with validation
    #######################################
    start_message = "TRAINING START"
    print("$" * (len(start_message) + 4))
    print("$ {} $".format(start_message))
    print("$" * (len(start_message) + 4))

    step = 0
    for epoch in range(config.num_epochs):
        print("Epoch [{epoch}/{num_epochs}]".format(epoch=(epoch+1), num_epochs=config.num_epochs))

        for train_batch in train_iter:
            # Validation
            if step % config.valid_freq == 0 or step % config.save_freq == 0:
                valid_dj_batch = valid_dijet_iter.next()
                valid_zj_batch = valid_zjet_iter.next()

                train_loss, train_acc = model.test_on_batch(
                    x=train_batch.x, y=train_batch.y)

                dijet_loss, dijet_acc = model.test_on_batch(
                    x=valid_dj_batch.x, y=valid_dj_batch.y)

                zjet_loss, zjet_acc = model.test_on_batch(
                    x=valid_zj_batch.x, y=valid_zj_batch.y)


                print("Step [{step}/{total_step}]".format(step=step, total_step=total_step))
                print("  Training:\n\tLoss {:.3f} | Acc. {:.3f}".format(train_loss, train_acc))
                print("  Validation on Dijet\n\tLoss {:.3f} | Acc. {:.3f}".format(dijet_loss, dijet_acc))
                print("  Validation on Z+jet\n\tLoss {:.3f} | Acc. {:.3f}".format(zjet_loss,zjet_acc))
                # print("  LR: {:.5f}".format(K.get_value(model.optimizer.lr)))


                meter.append({
                    "step": step, "lr": K.get_value(model.optimizer.lr),
                    "train_loss": train_loss, "dijet_loss": dijet_loss, "zjet_loss": zjet_loss,
                    "train_acc": train_acc, "dijet_acc": dijet_acc, "zjet_acc": zjet_acc})

            # Save model
            if (step != 0) and (step % config.save_freq == 0):
                filepath = os.path.join(
                    log_dir.saved_models.path,
                    "model_step-{step:06d}_loss-{loss:.3f}_acc-{acc:.3f}.h5".format(
                        step=step, loss=dijet_loss, acc=dijet_acc))
                _model.save(filepath)

            # Train on batch
            step += 1
            model.train_on_batch(
                x=train_batch.x, y=train_batch.y)
            # new_lr = np.power(step, -0.5)
            #K.set_value(_model.optimizer.lr, new_lr) 


        ###############################
        # On Epoch End
        ###########################

    #############################
    #
    #############################3
    filepath = os.path.join(log_dir.saved_models.path, "model_final.h5")
    _model.save(filepath)

    print("Training is over! :D")

    meter.add_plot(
        x="step",
        ys=[("train_loss", "Train/Dijet"),
            ("dijet_loss", "Validation/Dijet"),
            ("zjet_loss", "Validation/Z+jet")],
        title="Loss(CrossEntropy)", xlabel="Step", ylabel="Loss")

    meter.add_plot(
        x="step",
        ys=[("train_acc", "Train/Dijet"),
            ("dijet_acc", "Validation/Dijet"),
            ("zjet_acc", "Validation/Z+jet")],
        title="Accuracy", xlabel="Step", ylabel="Acc.")


    meter.finish()
    config.finish()
    
    return log_dir

if __name__ == "__main__":
    from evaluation import evaluate_all
    log_dir = train()
    evaluate_all(log_dir)


