from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import matplotlib
matplotlib.use('Agg')

import time
import argparse
import numpy as np
from datetime import datetime
import tensorflow as tf

import keras
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils import multi_gpu_model 

import keras.backend as K

from pipeline import DataLoader
# from pipeline import SeqDataLoader as DataLoader

sys.path.append("..")
from keras4jet.models import build_a_model
from keras4jet.meters import Meter
from keras4jet.utils import (
    get_log_dir,
    get_available_gpus,
    Logger
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="ak4")
    parser.add_argument("--directory", type=str, default="../../SJ-JetImage/step5/")
    parser.add_argument("--train_data", type=str, default="dijet")
    parser.add_argument(
	    '--log_dir', type=str,
	    default='./logs/{name}-{date}'.format(
            name="{name}", date=datetime.today().strftime("%Y-%m-%d_%H-%M-%S")),
	        help='the directory path of dataset')

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=len(get_available_gpus()))
    parser.add_argument("--train_batch_size", type=int, default=500)
    parser.add_argument("--val_batch_size", type=int, default=500)

    # Hyperparameter
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--maxlen", type=int, default=50)

    # Freq
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=500)

    args = parser.parse_args()

    train_data = args.directory+"/dijet_train.root"
    val_dijet_data = args.directory+"/dijet_test.root"
    val_zjet_data = args.directory+"/z_jet_test.root"

    if '{name}' in args.log_dir: args.log_dir = args.log_dir.format(name=args.model)
    log_dir = get_log_dir(path=args.log_dir,creation=True)

    logger = Logger(log_dir.path, "WRITE")
    logger.get_args(args)
    logger["train_data"] = train_data
    logger["val_dijet_data"] = val_dijet_data
    logger["val_zjet_data"] = val_zjet_data


    loss = 'binary_crossentropy'
    optimizer = optimizers.Adam(lr=args.lr)
    metric_list = ['accuracy']

    # data loader
    train_loader = DataLoader(
        path=train_data,
        maxlen=args.maxlen,
        batch_size=args.train_batch_size,
        cyclic=False)

    steps_per_epoch=np.ceil(len(train_loader)/train_loader.batch_size).astype(int)
    total_step = args.num_epochs * steps_per_epoch

    val_dijet_loader = DataLoader(
        path=val_dijet_data,
        maxlen=args.maxlen,
        batch_size=args.val_batch_size,
        cyclic=True)

    val_zjet_loader = DataLoader(
        path=val_zjet_data,
        maxlen=args.maxlen,
        batch_size=args.val_batch_size,
        cyclic=True)


    # build a model and compile it
    model = build_a_model(
        model_name=args.model,
        **train_loader.get_shapes())


    model.compile(
        loss=loss,
        optimizer=optimizers.Adam(lr=1e-2),
        metrics=['accuracy'])


    # Meter
    tr_acc_ = "train_{}_acc".format(args.train_data)
    tr_loss_ = "train_{}_loss".format(args.train_data)

    meter = Meter(
        data_name_list=[
            "step",
            tr_acc_, "val_dijet_acc", "val_zjet_acc",
            tr_loss_, "val_dijet_loss", "val_zjet_loss"],
        dpath=log_dir.validation.path)
    
    meter.prepare(
        data_pair_list=[("step", tr_acc_),
                        ("step", "val_dijet_acc"),
                        ("step", "val_zjet_acc")],
        title="Accuracy")

    meter.prepare(
        data_pair_list=[("step", tr_loss_),
                        ("step", "val_dijet_loss"),
                        ("step", "val_zjet_loss")],
        title="Loss(Cross-entropy)")


    # Training with validation
    step = 0
    start_time = time.time()
    for epoch in range(args.num_epochs):

        print("Epoch [{epoch}/{num_epochs}]".format(
            epoch=(epoch+1), num_epochs=args.num_epochs))


        for x_daus_train, x_glob_train, y_train in train_loader:

            # Validate model
            if step % args.val_freq == 0:
                x_daus_dijet, x_glob_dijet, y_dijet = val_dijet_loader.next()
                x_daus_zjet, x_glob_zjet, y_zjet = val_zjet_loader.next()

                train_loss, train_acc = model.test_on_batch(
                    x=[x_daus_train, x_glob_train], y=y_train)

                dijet_loss, dijet_acc = model.test_on_batch(
                    x=[x_daus_dijet, x_glob_dijet], y=y_dijet)

                zjet_loss, zjet_acc = model.test_on_batch(
                    x=[x_daus_zjet, x_glob_zjet], y=y_zjet)

                print("\nStep [{step}/{total_step}]".format(
                    step=step, total_step=total_step))

                print("  Training:")
                print("    Loss {train_loss:.3f} | Acc. {train_acc:.3f}".format(
                    train_loss=train_loss, train_acc=train_acc))

                print("  Validation on Dijet")
                print("    Loss {val_loss:.3f} | Acc. {val_acc:.3f}".format(
                    val_loss=dijet_loss, val_acc=dijet_acc))

                print("  Validation on Z+jet")
                print("    Loss {val_loss:.3f} | Acc. {val_acc:.3f}".format(
                    val_loss=zjet_loss, val_acc=zjet_acc))

                meter.append(data_dict={
                    "step": step,
                    tr_loss_: train_loss,
                    "val_dijet_loss": dijet_loss,
                    "val_zjet_loss": zjet_loss,
                    tr_acc_: train_acc,
                    "val_dijet_acc": dijet_acc,
                    "val_zjet_acc": zjet_acc})

            if (step!=0) and (step % args.save_freq == 0):
                filepath = os.path.join(
                    log_dir.saved_models.path,
                    "{name}_{step}.h5".format(name="model", step=step))
                model.save(filepath)


            # Train on batch
            model.train_on_batch(
                x=[x_daus_train, x_glob_train], y=y_train)
            step += 1
    end_time = time.time()
    logger["training_time"] = end_time - start_time
    print("Training is over! :D")
    filepath = os.path.join(log_dir.saved_models.path, "model_final.h5")
    model.save(filepath)
    meter.finish()
    logger.finish()
