from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import ROOT
import numpy as np
if __name__ == "__main__":
    import matplotlib as mpl
    mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# from tqdm import tqdm

import keras.backend as K
from keras.models import load_model

sys.path.append("..")
from keras4jet.data_loader import ImageSetLoader
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


def get_mean_stddev(path, image_list):
    root_file = ROOT.TFile.Open(path, "READ")
    root_file.ls()

    mean = []
    stddev = []
    for name in image_list:
        mu = root_file.Get("image_mean/{}".format(name))
        mu = np.array(mu)
        mean.append(mu)

        sigma = root_file.Get("image_stddev/{}".format(name))
        sigma = np.array(sigma)
        sigma[sigma == 0] += 1e-10

        stddev.append(sigma)


    mean = np.concatenate(mean).reshape(len(image_list), 33, 33)
    stddev = np.concatenate(stddev).reshape(len(image_list), 33, 33)


    return mean, stddev


def normalize_channel_wise(x):
    norm = x.sum(axis=(2, 3))
    norm = norm[:, :, np.newaxis, np.newaxis]
    norm[norm == 0] += 1e-10
    normalized_x = x / norm
    return normalized_x




def evaluate():
    log_fmt = "./logs/root-{}-{}"
    #train_fmt = "/scatch/slowmoyang/QGJets/root_{}_{}/3-JetImage/dijet_set/dijet_training_set.root"
    #test_fmt = "/scatch/slowmoyang/QGJets/root_{}_{}/3-JetImage/dijet_test_set.root"

    train_fmt = "../../Data/root_{}_{}/3-JetImage/dijet_set/dijet_training_set.root"
    test_fmt = "../../Data/root_{}_{}/3-JetImage/dijet_test_set.root"
    result_fmt = "./logs/auc_heatmap/result_model-{}_test-{}.npz" 

    for model_min_pt in range(100, 901, 100):
        model_max_pt = model_min_pt + 100
        print("Load a model")

        log_path = log_fmt.format(model_min_pt, model_max_pt)
        config = Config(log_path, mode="READ")

        best_model = get_best_model(log_path)
        print(best_model)
        model = load_model(best_model)

        train_path = train_fmt.format(model_min_pt, model_max_pt)
        mean, stddev = get_mean_stddev(train_path, config.x)

        for test_min_pt in range(100, 901, 100):
            test_max_pt = test_min_pt + 100
            test_set_path = test_fmt.format(test_min_pt, test_max_pt)

            data_loader = ImageSetLoader(
                path=test_set_path,
                x=config.x,
                x_shape=config.x_shape,
                extra=["pt", "eta"],
                batch_size=2048,
                cyclic=False)

            result_keys = ["y_true", "y_score", "pt", "eta"]
            result = {key: [] for key in result_keys}
            for idx, batch in enumerate(data_loader):
                x = batch["x"]
                x = normalize_channel_wise(x)
                x -= mean
                x /= stddev 

                result["y_true"].append(batch["y"])
                result["y_score"].append(model.predict_on_batch(batch["x"]))
                result["pt"].append(batch["pt"])
                result["eta"].append(batch["eta"])

                if idx % 5 == 0:
                    y_pred = np.where(result["y_score"][-1][:, 1] > 0.5, 1, 0)
                    correct = np.equal(result["y_true"][-1][:, 1], y_pred)
                    acc = correct.mean()
                    print("Accuracy: {}".format(acc))
            
            result = {key: np.concatenate(value) for key, value in result.iteritems()}
            result_path = result_fmt.format(model_min_pt, test_min_pt)
            np.savez(result_path, **result)

            del data_loader

        K.clear_session()


class Result(object):
    def __init__(self, path):
        result = np.load(path)
        for key in result.keys():
            setattr(self, key, result[key])
        y_pred = np.where(result["y_score"] > 0.5, 1, 0)
        setattr(self, "y_pred", y_pred)
        self.accuracy = metrics.accuracy_score(self.y_true, self.y_pred)
        self.roc_auc = metrics.roc_auc_score(self.y_true, self.y_score)
        
        name = os.path.splitext(os.path.split(path)[-1])[0]
        _, model, test_set = name.split("_")
        train_pt = int(model.split("-")[-1])
        test_pt = int(test_set.split("-")[-1])
        setattr(self, "train_pt", train_pt)
        setattr(self, "test_pt", test_pt)


def pt2idx(pt):
    return int(pt/100) - 1

def draw_auc_heatmap(result_dir="./logs/auc_heatmap"):
    result_paths = sorted(os.path.join(result_dir, each) for each in  os.listdir(result_dir))
    result_paths = [each for each in result_paths if os.path.isfile(each)]

    results = [Result(each) for each in result_paths]

    auc = np.zeros(shape=(9, 9))
    for each in results:
        train_idx = pt2idx(each.train_pt)
        test_idx = pt2idx(each.test_pt)
        auc[train_idx, test_idx] = each.roc_auc

    labels = ["{}\n~ {}\n".format(min_pt, min_pt+100) for min_pt in range(100, 1000, 100)]
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.heatmap(auc, ax=ax, vmax=0.9, vmin=0.7,
                annot=True, xticklabels=labels, yticklabels=labels, fmt=".3f", cmap=cmap)
    ax.set_title("ConvNet AUC", fontsize=20)
    ax.set_xlabel("Training Set pT Range [GeV]", fontsize=15)
    ax.set_ylabel("Test Set pT Range [GeV]", fontsize=15)
    ax.invert_yaxis()

    path_fmt = os.path.join(result_dir, "auc_heatmap.{ext}")
    np.save(path_fmt.format(ext="npy"), auc)
    fig.savefig(path_fmt.format(ext="png"))
    fig.savefig(path_fmt.format(ext="pdf"), format="pdf")



if __name__ == "__main__":
    # evaluate()
    draw_auc_heatmap()
