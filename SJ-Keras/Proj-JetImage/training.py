from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

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
from keras4jet.models import build_a_model
from keras4jet.meters import Meter
from keras4jet.utils import (
    get_log_dir,
    Logger,
    get_available_gpus
)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, default="dijet")
    parser.add_argument("--directory", type=str, default="../../SJ-JetImage/image33x33/")
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument('--log_dir', type=str,
	                default='./logs/{name}-{date}'.format(
                            name="{name}",
                            date=datetime.today().strftime("%Y-%m-%d_%H-%M-%S")))

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=len(get_available_gpus()))
    parser.add_argument("--train_batch_size", type=int, default=500)
    parser.add_argument("--val_batch_size", type=int, default=500)
    parser.add_argument("--multi-gpu", default=False, action='store_true', dest='multi_gpu')

    # Hyperparameter
    parser.add_argument("--lr", type=float, default=0.001)

    # Freq
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=500)

    args = parser.parse_args()

    if args.train_data == "dijet":
        train_data = args.directory+"/dijet_train.root"
        val_dijet_data = args.directory+"/dijet_test_after_dijet.root"
        val_zjet_data = args.directory+"/z_jet_test_after_dijet.root"
    elif args.train_data == "zjet":
        train_data = args.directory+"/z_jet_train.root"
        val_dijet_data = args.directory+"/dijet_test_after_zjet.root"
        val_zjet_data = args.directory+"/z_jet_test_after_zjet.root"
    else:
        raise ValueError("")

    if '{name}' in args.log_dir: args.log_dir = args.log_dir.format(name=args.model)
    log_dir = get_log_dir(path=args.log_dir, creation=True)

    logger = Logger(dpath=log_dir.path, mode="WRITE")
    logger.get_args(args)
    logger["train_data"] = train_data
    logger["val_zjet_data"] = val_zjet_data
    logger["val_dijet_data"] = val_dijet_data

    # data loader
    train_loader = DataLoader(
        path=train_data,
        batch_size=args.train_batch_size,
        cyclic=False)

    steps_per_epoch = np.ceil( len(train_loader) / train_loader.batch_size ).astype(int)
    total_step = args.num_epochs * steps_per_epoch

    val_dijet_loader = DataLoader(
        path=val_dijet_data,
        batch_size=args.val_batch_size,
        cyclic=True)

    val_zjet_loader = DataLoader(
        path=val_zjet_data,
        batch_size=args.val_batch_size,
        cyclic=True)


    loss = 'binary_crossentropy'
    optimizer = optimizers.Adam(lr=args.lr)
    metric_list = ['accuracy']

    # build a model and compile it
    _model = build_a_model(model_name=args.model, input_shape=train_loader._image_shape)

    if args.multi_gpu:
        model = multi_gpu_model(_model, gpus=args.num_gpus)
    else:
        model = _model


    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metric_list
    )


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
    for epoch in range(args.num_epochs):

        print("Epoch [{epoch}/{num_epochs}]".format(
            epoch=(epoch+1), num_epochs=args.num_epochs))


        for x_train, y_train in train_loader:

            # Validate model
            if step % args.val_freq == 0:
                x_dijet, y_dijet = val_dijet_loader.next()
                x_zjet, y_zjet = val_zjet_loader.next()

                train_loss, train_acc = model.test_on_batch(x=x_train, y=y_train)
                dijet_loss, dijet_acc = model.test_on_batch(x=x_dijet, y=y_dijet)
                zjet_loss, zjet_acc = model.test_on_batch(x=x_zjet, y=y_zjet)

                print("Step [{step}/{total_step}]".format(
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
                _model.save(filepath)


            # Train on batch
            model.train_on_batch(x=x_train, y=y_train)
            step += 1

    filepath = os.path.join(log_dir.saved_models.path,
                            "model_final.h5")
    _model.save(filepath)
    print("Training is over! :D")
    meter.finish()
    logger.finish()
    

            





if __name__ == "__main__":
 main()
