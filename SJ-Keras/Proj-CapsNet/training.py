from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import matplotlib
matplotlib.use('Agg')

import argparse
import numpy as np
from datetime import datetime

import keras
from keras import optimizers
from keras import losses
from keras import metrics
from keras.utils import multi_gpu_model 

import keras.backend as K

from pipeline import DataLoader

sys.path.append("..")
from keras4jet.models import build_a_model
from keras4jet.losses import margin_loss
from keras4jet.losses import reco_loss
from keras4jet.meters import Meter
from keras4jet import train_utils
from keras4jet.utils import get_log_dir
from keras4jet.utils import Logger
from keras4jet.utils import get_available_gpus
from keras4jet.utils import get_dataset_paths

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, default="dijet")
    parser.add_argument("--directory", type=str,
                        default="../../Data/pt_100_500/")
    parser.add_argument("--model", type=str, default="capsnet")
    parser.add_argument('--log_dir', type=str,
        default='./logs/{name}-{date}'.format(
            name="{name}", date=datetime.today().strftime("%Y-%m-%d_%H-%M-%S")))

    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_gpus", type=int, default=len(get_available_gpus()))
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--val_batch_size", type=int, default=500)
    parser.add_argument("--multi-gpu", default=False, action='store_true', dest='multi_gpu')

    # Hyperparameter
    parser.add_argument("--lr", type=float, default=0.001)

    # Freq
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=500)

    # 
    
    
    args = parser.parse_args()

    if '{name}' in args.log_dir:
        args.log_dir = args.log_dir.format(name="CapsNet")

    log_dir = get_log_dir(path=args.log_dir, creation=True)

    dataset_paths = get_dataset_paths(args.directory, args.train_data)

    logger = Logger(dpath=log_dir.path, mode="WRITE")
    logger.update(args)
    logger.update(dataset_paths)

    # data loader
    logger.update({
        "x": ["image_cpt_33", "image_npt_33", "image_cmult_33"],
        "x_shape": (3, 33, 33)})

    train_loader = DataLoader(
        path=logger["training_set"],
        x=logger["x"],
        x_shape=logger["x_shape"],
        batch_size=args.batch_size,
        cyclic=False)

    steps_per_epoch = np.ceil( len(train_loader) / train_loader.batch_size ).astype(int)
    total_step = args.num_epochs * steps_per_epoch

    val_dijet_loader = DataLoader(
        path=logger["dijet_validation_set"],
        x=logger["x"],
        x_shape=logger["x_shape"],
        batch_size=args.val_batch_size,
        cyclic=True)

    val_zjet_loader = DataLoader(
        path=logger["zjet_validation_set"],
        x=logger["x"],
        x_shape=logger["x_shape"],
        batch_size=args.val_batch_size,
        cyclic=True)

    # build a model and compile it
    _model = build_a_model(
        model_name=args.model,
        input_shape=logger["x_shape"],
        num_classes=2,
        num_routings=3)

    if args.multi_gpu:
        model = multi_gpu_model(_model, gpus=args.num_gpus)
    else:
        model = _model

    model.compile(
        loss=[margin_loss, reco_loss],
        loss_weights=[1.0, 0.0005],
        optimizer=optimizers.Adam(lr=args.lr),
        metrics=["accuracy"])

    # Meter
    meter = Meter(
        name_list=["step", "lr",
         "train_loss", "dijet_loss", "zjet_loss",
         "train_margin_loss", "dijet_margin_loss", "zjet_margin_loss",
         "train_reco_loss", "dijet_reco_loss", "zjet_reco_loss",
         "train_acc", "dijet_acc", "zjet_acc"],
        dpath=log_dir.validation.path)
                   
    lr_scheduler = train_utils.ReduceLROnPlateau(model)

    # Training with validation
    step = 0
    for epoch in range(args.num_epochs):

        print("Epoch [{epoch}/{num_epochs}]".format(
            epoch=(epoch+1), num_epochs=args.num_epochs))

        val_loss_accum = 0.0
        num_val_steps = 0

        for x_train, y_train in train_loader:
            # Validate model
            if step % args.val_freq == 0:
                x_dijet, y_dijet = val_dijet_loader.next()
                x_zjet, y_zjet = val_zjet_loader.next()

                # [Loss, MarginLoss, RecoLoss, Acc, RecoAcc]
                train_metrics = model.test_on_batch(
                    x=[x_train, y_train], y=[y_train, x_train])
                dijet_metrics = model.test_on_batch(
                    x=[x_dijet, y_dijet], y=[y_dijet, x_dijet])
                zjet_metrics = model.test_on_batch(
                    x=[x_zjet, y_zjet], y=[y_zjet, x_zjet])

                val_loss_accum += dijet_metrics[0]
                num_val_steps += 1

                print("Step [{step}/{total_step}]".format(
                    step=step, total_step=total_step))

                print("  Training:")
                print("    Loss {train_loss:.3f} | Acc. {train_acc:.3f}".format(
                    train_loss=train_metrics[0], train_acc=train_metrics[3]))

                print("  Validation on Dijet")
                print("    Loss {val_loss:.3f} | Acc. {val_acc:.3f}".format(
                    val_loss=dijet_metrics[0], val_acc=dijet_metrics[3]))

                print("  Validation on Z+jet")
                print("    Loss {val_loss:.3f} | Acc. {val_acc:.3f}".format(
                    val_loss=zjet_metrics[0], val_acc=zjet_metrics[0]))

                # [Loss, MarginLoss, RecoLoss, Acc, RecoAcc]
                meter.append(data_dict={
                    "step": step,
                    "lr":K.get_value(model.optimizer.lr),
                    # train metrics
                    "train_loss": train_metrics[0],
                    "train_margin_loss": train_metrics[1],
                    "train_reco_loss": train_metrics[2],
                    "train_acc": train_metrics[3],
                    # dijet
                    "dijet_loss": dijet_metrics[0],
                    "dijet_margin_loss": dijet_metrics[1],
                    "dijet_reco_loss": dijet_metrics[2],
                    "dijet_acc": dijet_metrics[3],
                    # zjet
                    "zjet_loss": zjet_metrics[0],
                    "zjet_margin_loss": zjet_metrics[1],
                    "zjet_reco_loss": zjet_metrics[2],
                    "zjet_acc": zjet_metrics[3]})

            if (step!=0) and (step % args.save_freq == 0):
                filepath = os.path.join(
                    log_dir.saved_models.path,
                    "{name}_{step}.h5".format(name="model", step=step))
                _model.save(filepath)


            # Train on batch
            model.train_on_batch(
                x=[x_train, y_train], y=[y_train, x_train])

            step += 1

        # End of the epoch
        avg_val_loss = val_loss_accum / num_val_steps
        lr_scheduler.step(metrics=avg_val_loss, epoch=epoch)

    # End of the training
    filepath = os.path.join(
        log_dir.saved_models.path, "model_final.h5")
    _model.save(filepath)

    print("Training is over! :D")

    meter.add_plot(
        x="step",
        ys=[("train_loss", "Train/Dijet"),
            ("dijet_loss", "Validation/Dijet"),
            ("zjet_loss", "Validation/Z+jet")],
        title="Loss", xlabel="Step", ylabel="Loss")

    meter.add_plot(
        x="step",
        ys=[("train_marin_loss", "Train/Dijet"),
            ("dijet_marin_loss", "Validation/Dijet"),
            ("zjet_margin_loss", "Validation/Z+jet")],
        title="Margin Loss", xlabel="Step", ylabel="Loss")

    meter.add_plot(
        x="step",
        ys=[("train_reco_loss", "Train/Dijet"),
            ("dijet_reco_loss", "Validation/Dijet"),
            ("zjet_reco_loss", "Validation/Z+jet")],
        title="Reconstruction Loss", xlabel="Step", ylabel="Loss")


    meter.add_plot(
        x="step",
        ys=[("train_acc", "Train/Dijet"),
            ("dijet_acc", "Validation/Dijet"),
            ("zjet_acc", "Validation/Z+jet")],
        title="Accuracy", xlabel="Step", ylabel="Acc.")

    meter.add_plot(
        x="step",
        ys=[("lr", "Train/Dijet", "indianred")],
        title="Learning Rate", xlabel="Step", ylabel="LR")

    meter.finish()
    logger.finish()
    
    return log_dir


if __name__ == "__main__":
    from evaluation import evaluate_all
    log_dir = train()
    evaluate_all(log_dir)
