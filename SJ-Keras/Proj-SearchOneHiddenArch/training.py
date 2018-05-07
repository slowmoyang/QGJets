from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib as mpl
mpl.use("Agg")

import argparse
import numpy as np
from datetime import datetime

import tensorflow as tf

import keras
import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils import multi_gpu_model 

sys.path.append("..")
from keras4jet.models import build_a_model
from keras4jet.data_loader import FeatureLoader
from keras4jet.meters import Meter
from keras4jet.metrics_wrapper import roc_auc_score
from keras4jet import train_utils
from keras4jet.utils import get_log_dir
from keras4jet.utils import Config
from keras4jet.utils import get_available_gpus
from keras4jet.utils import get_dataset_paths

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_sample", default="dijet", type=str)
    parser.add_argument("--datasets_dir",
                        default="/data/slowmoyang/QGJets/root_100_200/3-JetImage/",
                        type=str)
    parser.add_argument("--model", default="dnn", type=str)

    parser.add_argument("--log_dir", default="./logs/{name}", type=str)
    parser.add_argument("--num_gpus", default=len(get_available_gpus()), type=int)
    parser.add_argument("--multi-gpu", default=False, action='store_true', dest='multi_gpu')

    # Hyperparameters
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--val_batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    # Frequencies
    parser.add_argument("--val_freq", type=int, default=32)
    parser.add_argument("--save_freq", type=int, default=32)

    # Project parameters
    parser.add_argument("--hidden_units", type=int, default=100)
    parser.add_argument("--x", nargs="+", default=["cmult", "nmult", "axis1", "axis2", "ptD"])

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

    dataset_paths = get_dataset_paths(config.datasets_dir, config.train_sample)
    config.update(dataset_paths)

    ########################################
    # Load training and validation datasets
    ########################################

    train_loader = FeatureLoader(
        path=config.training_set,
        x=config.x,
        batch_size=config.batch_size,
        cyclic=False)

    steps_per_epoch = int(len(train_loader) / train_loader.batch_size)
    total_step = config.num_epochs * steps_per_epoch

    val_dijet_loader = FeatureLoader(
        path=config.dijet_validation_set,
        x=config.x,
        batch_size=config.val_batch_size,
        cyclic=True)

    val_zjet_loader = FeatureLoader(
        path=config.zjet_validation_set,
        x=config.x,
        batch_size=config.val_batch_size,
        cyclic=True)


    #################################
    # Build & Compile a model.
    #################################
    config["model_type"] = "features"

    _model = build_a_model(
        model_type=config.model_type,
        model_name=config.model,
        num_features=len(config.x),
        units_list=[config.hidden_units])
        

    if config.multi_gpu:
        model = multi_gpu_model(_model, gpus=config.num_gpus)
    else:
        model = _model

    # TODO config should have these information.
    loss = 'categorical_crossentropy'
    optimizer = optimizers.Adam(lr=config.lr)
    metric_list = ['accuracy', roc_auc_score]

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metric_list)

    lr_scheduler = train_utils.ReduceLROnPlateau(model)

    #######################################
    # 
    ###########################################

    meter = Meter(
        name_list=["step", "lr",
                   "train_loss", "dijet_loss", "zjet_loss",
                   "train_acc", "dijet_acc", "zjet_acc",
                   "train_auc", "dijet_auc", "zjet_auc"],
        dpath=log_dir.validation.path)

    #######################################
    # Training with validation
    #######################################
    step = 0
    for epoch in range(config.num_epochs):
        print("Epoch [{epoch}/{num_epochs}]".format(epoch=(epoch+1), num_epochs=config.num_epochs))

        for train_batch in train_loader:
            # Validation
            if step % config.val_freq == 0 or step % config.save_freq == 0:
                val_dj_batch = val_dijet_loader.next()
                val_zj_batch = val_zjet_loader.next()

                train_loss, train_acc, train_auc = model.test_on_batch(x=train_batch["x"], y=train_batch["y"])
                dijet_loss, dijet_acc, dijet_auc = model.test_on_batch(x=val_dj_batch["x"], y=val_dj_batch["y"])
                zjet_loss, zjet_acc, zjet_auc = model.test_on_batch(x=val_zj_batch["x"], y=val_zj_batch["y"])

                lr_scheduler.monitor(metrics=dijet_loss)

                print("Step [{step}/{total_step}]".format(step=step, total_step=total_step))
                print("  Training:\n\tLoss {:.3f} | Acc. {:.3f} | AUC {:.3f}".format(train_loss, train_acc, train_auc))
                print("  Validation on Dijet\n\tLoss {:.3f} | Acc. {:.3f} | AUC {:.3f}".format(dijet_loss, dijet_acc, dijet_auc))
                print("  Validation on Z+jet\n\tLoss {:.3f} | Acc. {:.3f} | AUC {:.3f}".format(zjet_loss,zjet_acc, zjet_auc))

                meter.append({
                    "step": step, "lr": K.get_value(model.optimizer.lr),
                    "train_loss": train_loss, "dijet_loss": dijet_loss, "zjet_loss": zjet_loss,
                    "train_acc": train_acc, "dijet_acc": dijet_acc, "zjet_acc": zjet_acc,
                    "train_auc": train_auc, "dijet_auc": dijet_auc, "zjet_auc": zjet_auc})

            # Save model
            if (step != 0) and (step % config.save_freq == 0):
                filepath = os.path.join(
                    log_dir.saved_models.path,
                    "model_step-{step:06d}_loss-{loss:.3f}_acc-{acc:.3f}_auc-{auc:.3f}.h5".format(
                        step=step, loss=dijet_loss, acc=dijet_acc, auc=dijet_auc))
                _model.save(filepath)

            # Train on batch
            model.train_on_batch(x=train_batch["x"], y=train_batch["y"])
            step += 1

        ###############################
        # On Epoch End
        ###########################
        lr_scheduler.step(epoch=epoch)

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

    meter.add_plot(
        x="step",
        ys=[("train_auc", "Train/Dijet"),
            ("dijet_auc", "Validation/Dijet"),
            ("zjet_auc", "Validation/Z+jet")],
        title="AUC", xlabel="Step", ylabel="AUC")



    meter.finish()
    config.finish()
    
    return log_dir

if __name__ == "__main__":
    from evaluation import evaluate_all
    log_dir = train()
    evaluate_all(log_dir)


