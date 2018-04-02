from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

from tqdm import tqdm

import keras.backend as K
from keras.models import load_model

sys.path.append("/home/slowmoyang/Lab/QGJets/Keras")
from keras4jet.data_loader import ImageDataLoader
from keras4jet.utils import get_log_dir
from keras4jet.utils import Config


def parse_roc_path(path):
    path = os.path.split(path)[-1]
    path = os.path.splitext(path)[0]
    process, _, step, auc = path.split("_")
    step = step.split("-")[-1]
    if step.isdigit():
        step = int(step)
    auc = float(auc.split("-")[-1])
    output = {"process": process, "step": step, "auc": auc}
    return output


def parse_saved_model_path(path):
    name = os.path.split(path)[-1]
    name = os.path.splitext(name)[0]
    step = name.split("_")[-1]
    if step.isdigit():
        step = int(step)
    output = {"path": path, "step": step}
    return output

def get_best_model(log_dir_path):
    log_dir = get_log_dir(log_dir_path, creation=False)
    config = Config(log_dir_path, mode="READ")

    roc_entries = log_dir.roc.get_entries()
    roc_entries = [each for each in roc_entries if os.path.splitext(each)[-1] == ".csv"]
    roc_entries = [parse_roc_path(each) for each in roc_entries]

    #roc_entries = filter(lambda each: each[0] == config.train_sample, roc_entries)
    roc_entries = [each for each in roc_entries if each["process"] == config.train_sample]
    best_step = max(roc_entries, key=lambda each: each["auc"])["step"]

    model_entries = log_dir.saved_models.get_entries()
    # model_entries = map(parse_saved_model_path, model_entries)
    model_entries = [parse_saved_model_path(each) for each in model_entries]
    # best_model = filter(lambda each: each[1] == best_step, model_entries)
    print(model_entries)
    best_model = [each["path"] for each in model_entries if each["step"] == best_step]

    if len(best_model) > 1:
        print("")

    return best_model[0]



def main():
    result_fmt = "./logs/result_model-{}_test-{}.npz"

    for model_min_pt in range(200, 901, 100):
        model_max_pt = model_min_pt + 100
        print("Load a model trained on {} to {} GeV".format(model_min_pt, model_max_pt))

        log_dir_path = "./logs/root_{}_{}".format(model_min_pt, model_max_pt)
        # log_dir = get_log_dir(log_dir_path, creation=False)
        config = Config(log_dir_path, mode="READ")

        best_model = get_best_model(log_dir_path) 
        print(best_model)
        model = load_model(best_model)
        
        # for test_min_pt in range(200, 901, 100):
        for test_min_pt in [100]:
            test_max_pt = test_min_pt + 100
            print("Test the model on {} ~ {} GeV".format(test_min_pt, test_max_pt))
            test_set_path = "../../Data/root_{}_{}/3-JetImage/dijet_set/dijet_test_set.root".format(
                test_min_pt, test_max_pt)

            data_loader = ImageDataLoader(
                path=test_set_path,
                x=config.x,
                x_shape=config.x_shape,
                extra=["pt", "eta"],
                batch_size=2048,
                cyclic=False)

            y_true = []
            y_score = []
            pt = []
            eta = []

            for idx, batch in enumerate(data_loader):
                y_true.append(batch["y"])
                y_score.append(model.predict_on_batch(batch["x"]))
                pt.append(batch["pt"])
                eta.append(batch["eta"])

                if idx % 5 == 0:
                    y_pred = np.where(y_score[-1][:, 1] > 0.5, 1, 0)
                    correct = np.equal(y_true[-1][:, 1], y_pred)
                    acc = correct.mean()
                    print("Accuracy: {}".format(acc))
                



            y_true = np.concatenate(y_true)
            y_score = np.concatenate(y_score)
            pt = np.concatenate(pt)
            eta = np.concatenate(eta)

            np.savez(result_fmt.format(model_min_pt, test_min_pt), y_true=y_true, y_score=y_score, pt=pt, eta=eta)

            del data_loader

        K.clear_session()



if __name__ == "__main__":
    main()
