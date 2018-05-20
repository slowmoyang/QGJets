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

import keras
import keras.backend as K
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils import multi_gpu_model 

sys.path.append("..")
from keras4jet.models import build_a_model
from keras4jet.data_loader import HybridIFLoader
from keras4jet.meters import Meter
from keras4jet import train_utils
from keras4jet.utils import get_log_dir
from keras4jet.utils import Config
from keras4jet.utils import get_available_gpus
from keras4jet.utils import get_dataset_paths


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_sample", default="dijet", type=str)
    parser.add_argument("--datasets_dir",
                        default="/scatch/slowmoyang/QGJets/root_100_200/3-JetImage",
                        type=str)
    parser.add_argument("--model", default="image_plus_features", type=str)


    parser.add_argument("--log_dir", default="./logs/{name}", type=str)
    parser.add_argument("--num_gpus", default=len(get_available_gpus()), type=int)
    parser.add_argument("--multi-gpu", default=False, action='store_true', dest='multi_gpu')

    # Hyperparameters
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--val_batch_size", default=2048, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    # Frequencies
    parser.add_argument("--val_freq", type=int, default=32)
    parser.add_argument("--save_freq", type=int, default=250)

    # Project parameters
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--features", nargs="+", default=["axis1", "axis2", "cmult", "nmult", "ptD"])

    args = parser.parse_args()

    # Log directory
    if '{name}' in args.log_dir:
        args.log_dir = args.log_dir.format(
            name="Untitled_{}".format(
                datetime.today().strftime("%Y-%m-%d_%H-%M-%S")))
    log_dir = get_log_dir(path=args.log_dir, creation=True)
 
    # Config
    config = Config(dpath=log_dir.path, mode="WRITE")
    config.update(args)

    ########################################
    # Load training and validation datasets
    ########################################
    dataset_paths = get_dataset_paths(config.datasets_dir, config.train_sample)
    config.update(dataset_paths)

    config["image"] = [
        "image_chad_pt_33", "image_chad_mult_33",
        "image_electron_pt_33", "image_electron_mult_33",
        "image_muon_pt_33", "image_muon_mult_33",
        "image_nhad_pt_33", "image_nhad_mult_33",
        "image_photon_pt_33", "image_photon_mult_33"]

    config["image_shape"] = (len(config.image), 33, 33)

    train_loader = HybridIFLoader(
        path=config.training_set,
        features=config.features,
        image=config.image,
        image_shape=config.image_shape,
        batch_size=config.batch_size,
        cyclic=False)

    steps_per_epoch = int(len(train_loader) / train_loader.batch_size)
    total_step = config.num_epochs * steps_per_epoch
    config["steps_per_epoch"] = steps_per_epoch
    config["total_step"] = total_step

    val_dijet_loader = HybridIFLoader(
        path=config.dijet_validation_set,
        features=config.features,
        image=config.image,
        image_shape=config.image_shape,
        batch_size=config.val_batch_size,
        cyclic=True)

    val_zjet_loader = HybridIFLoader(
        path=config.zjet_validation_set,
        features=config.features,
        image=config.image,
        image_shape=config.image_shape,
        batch_size=config.val_batch_size,
        cyclic=True)


    #################################
    # Build & Compile a model.
    #################################
    config["model_type"] = "hybrid"

    config["units_list"] = [16, 32, 64]
    config["filters_list"] = [16, 32, 32, 64, 64]

    _model = build_a_model(
        model_type=config.model_type,
        model_name=config.model,
        image_shape=config.image_shape,
        num_features=len(config.features),
        units_list=config.units_list,
        filters_list=config.filters_list,
        kernel_size=config.kernel_size,
        padding="SAME")

    if config.multi_gpu:
        model = multi_gpu_model(_model, gpus=config.num_gpus)
    else:
        model = _model

    # TODO config should have these information.
    loss = 'categorical_crossentropy'
    optimizer = optimizers.Adam(lr=config.lr)
    metric_list = ['accuracy']

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
                    "train_acc", "dijet_acc", "zjet_acc"],
        dpath=log_dir.validation.path)

    #######################################
    # Training with validation
    #######################################
    step = 0
    for epoch in range(config.num_epochs):
        print("Epoch [{epoch}/{num_epochs}]".format(
            epoch=(epoch+1), num_epochs=config.num_epochs))

        for train_batch in train_loader:

            # Validate model
            if step % config.val_freq == 0 or step % config.save_freq == 0:
                val_dj_batch = val_dijet_loader.next()
                val_zj_batch = val_zjet_loader.next()

                train_loss, train_acc = model.test_on_batch(
                    x=[train_batch["image"], train_batch["features"]],
                    y=train_batch["y"])

                dijet_loss, dijet_acc = model.test_on_batch(
                    x=[val_dj_batch["image"], val_dj_batch["features"]],
                    y=val_dj_batch["y"])

                zjet_loss, zjet_acc = model.test_on_batch(
                    x=[val_zj_batch["image"], val_zj_batch["features"]],
                    y=val_zj_batch["y"])

                lr_scheduler.monitor(metrics=dijet_loss)

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

                meter.append({
                    "step": step,
                    "lr": K.get_value(model.optimizer.lr),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "dijet_loss": dijet_loss,
                    "dijet_acc": dijet_acc,
                    "zjet_loss": zjet_loss,
                    "zjet_acc": zjet_acc})

            if (step != 0) and (step % config.save_freq == 0):
                filepath = os.path.join(
                    log_dir.saved_models.path,
                    "model_step-{step:06d}_loss-{loss:.3f}_acc-{acc:.3f}.h5".format(
                        step=step, loss=dijet_loss, acc=dijet_acc))
                _model.save(filepath)

            # Train on batch
            model.train_on_batch(
                x=[train_batch["image"], train_batch["features"]],
                y=train_batch["y"])
            step += 1

        # FIXME
        lr_scheduler.step(epoch=epoch)

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

