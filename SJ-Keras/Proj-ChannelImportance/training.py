from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path

import matplotlib
matplotlib.use('Agg')

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

sys.path.append("..")
from keras4jet.models.image.resnet import build_a_model
from keras4jet.meters import Meter
from keras4jet.utils import (
    get_log_dir,
    Logger,
    get_available_gpus
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, default="dijet")
    parser.add_argument(
        '--log_dir', type=str,
	    default='./logs/{name}-{date}'.format(
        name="name", date=datetime.today().strftime("%Y-%m-%d_%H-%M-%S")))

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=len(get_available_gpus()))
    parser.add_argument("--train_batch_size", type=int, default=500)
    parser.add_argument("--val_batch_size", type=int, default=500)

    # Hyperparameter
    parser.add_argument("--lr", type=float, default=0.001)

    # Freq
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=500)

    ##
    parser.add_argument("--channel", type=str, default="cpt")

    args = parser.parse_args()

    data_dir = "../../SJ-JetImage/images33x33/"
    if args.train_data == "dijet":
        train_data = os.path.join(data_dir, "")
    elif args.train_data == "zjet":
        train_data = os.path.join(data_dir, "")
    else:
        raise ValueError("")

    log_dir = get_log_dir(
        path=args.log_dir.format(name=args.channel),
        creation=True)

    logger = Logger(dpath=log_dir.path, mode="WRITE")
    logger.get_args(args)
    logger["train_data"] = train_data
    logger["val_zjet_data"] = val_dijet_data
    logger["val_dijet_data"] = val_dijet_data

    # data loader
    # data loader for training data
    train_loader = DataLodaer(
        path=train_data,
        batch_size=args.train_batch_size,
        cyclic=False)

    steps_per_epoch = np.ceil(
        len(train_loader) / train_loader.batch_size).astype(int)
    total_step = args.num_epochs * steps_per_epoch

    # data loaders for dijet/z+jet validation data
    val_dijet_loader = DataLodaer(
        path=val_dijet_data,
        batch_size=args.val_batch_size,
        cyclic=True)

    val_zjet_loader = DataLodaer(
        path=val_zjet_data,
        batch_size=args.val_batch_size,
        cyclic=True)

    # build a model
    _model = build_a_model(input_shape)
    model = multi_gpu_model(_model, gpus=args.num_gpus) 


    # Define 
    loss = 'binary_crossentropy'
    optimizer = optimizers.Adam(lr=args.lr)
    metrics = ['accuracy']

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)


    # Meter
    tr_acc_ = "train_{}_acc".format(args.train_data)
    tr_loss_ = "train_{}_loss".format(args.train_data)

    meter = Meter(
        data_name_list=[
            "step",
            tr_acc_, "val_acc_dijet", "val_acc_zjet",
            tr_loss_, "val_loss_dijet", "val_loss_zjet"],
        dpath=log_dir.validation.path)
    


    # Training with validation
    step = 0
    for epoch in range(args.num_epochs):

        print("Epoch [{epoch}/{num_epochs}]".format(
            epoch=(epoch+1), num_epochs=args.num_epochs))

        for x_train, y_train in train_loader:

            # Validate model
            if step % args.val_freq == 0:
                x_dijet, y_dijet = val_dijet_loader.next()
                x_zjet, y_zjet = val_zjet_loader.next()

                loss_train, train_acc = model.test_on_batch(x=x_train, y=y_train)
                loss_dijet, acc_dijet = model.test_on_batch(x=x_dijet, y=y_dijet)
                loss_zjet, acc_zjet = model.test_on_batch(x=x_zjet, y=y_zjet)

                print("Step [{step}/{total_step}]".format(
                    step=step, total_step=total_step))

                print("  Training:")
                print("    Loss {loss_train:.3f} | Acc. {train_acc:.3f}".format(
                    loss_train=loss_train, train_acc=train_acc))

                print("  Validation on Dijet")
                print("    Loss {val_loss:.3f} | Acc. {val_acc:.3f}".format(
                    val_loss=loss_dijet, val_acc=acc_dijet))

                print("  Validation on Z+jet")
                print("    Loss {val_loss:.3f} | Acc. {val_acc:.3f}".format(
                    val_loss=loss_zjet, val_acc=acc_zjet))

                meter.append(data_dict={
                    "step": step,
                    tr_loss_: loss_train,
                    "val_loss_dijet": loss_dijet,
                    "val_loss_zjet": loss_zjet,
                    tr_acc_: train_acc,
                    "val_acc_dijet": acc_dijet,
                    "val_acc_zjet": acc_zjet})

            # Save model
            if (step!=0) and (step % args.save_freq == 0):
                filepath = os.path.join(
                    log_dir.saved_models.path,
                    "{name}_{step}.h5".format(name="model", step=step))
                _model.save(filepath)


            # Train on batch
            model.train_on_batch(x=x_train, y=y_train)
            step += 1

    print("Training is over! :D")

    meter.prepare(
        x="step",
        ys=[(tr_acc_, "Training Acc on {}"),
            ("val_acc_dijet", "Validation on Dijet"),
            ("val_acc_zjet", "Validation on Z+jet")],
        title="Accuracy", xaxis="Step", yaxis="Accuracy")

    meter.prepare(
        x="step",
        ys=[(tr_loss_name, "Training Loss os {}"),
            ("val_acc_dijet", "Validation Loss on Dijet"),
            ("val_acc_zjet", "Validation Loss on Z+jet")],
        title="Loss", xaxis="Step", yaxis="Accuracy")

    meter.finish()
    logger.finish()
    

            





if __name__ == "__main__":
 main()
