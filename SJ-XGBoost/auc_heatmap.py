from __future__ import division

import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt

import os
import xgboost as xgb
import numpy as np
from sklearn import metrics
import argparse
import seaborn as sns

from data_loading import load_dataset
from utils import Config

def load_classifier(path):
    booster = xgb.Booster()
    booster.load_model(path)
    classifier = xgb.XGBClassifier()
    classifier._Booster = booster
    return classifier

def parse_model_path(path):
    print(path)
    _, min_pt, max_pt = path.split("/")[-2].split("_")
    return min_pt, max_pt

def parse_dataset_path(path):
    dname, fname = path.split("/")[-2:]
    _, min_pt, max_pt = dname.split("_")
    process, _, _ = fname.split("_")
    min_pt = int(min_pt)
    max_pt = int(max_pt)
    return (process, min_pt, max_pt)

def main(log_dir, dataset_dir, out_dir):
    models = os.listdir(log_dir)
    config_path = os.path.join(log_dir, models[0])
    config = Config(config_path, "READ")

    model_path_list = [os.path.join(log_dir, each, "xgb.model") for each in models]

    dataset_fmt = "/data/slowmoyang/QGJets/npz/root_{}_{}/dijet_test_set.npz"
    datasets = [dataset_fmt.format(min_pt, min_pt+100) for min_pt in range(100, 901, 100)]

    auc_matrix = [[] for min_pt in range(100, 1000, 100)]

    for model_idx, model_path in enumerate(model_path_list):
        clf = load_classifier(model_path)

        model_min_pt, _ = parse_model_path(model_path)
        for dataset_path in datasets:
            _, dataset_min_pt, _ = parse_dataset_path(dataset_path)
            x, y_true, _ = load_dataset(dataset_path, features=config.feature_names)

            y_score = clf.predict_proba(x)[:, 1]
            auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
            auc_matrix[model_idx].append(auc)

    auc_mat = np.array(auc_matrix, np.float32)
    print(auc_mat)

    path_fmt = os.path.join(out_dir, "auc_heatmap.{ext}")
    np.save(path_fmt.format(ext="npy"), auc_mat)

    sns.set(font_scale=1.2)
    fig, ax = plt.subplots()

    fig.set_figheight(8)
    fig.set_figwidth(12)

    labels = ["{}\n~ {}\n".format(min_pt, min_pt+100) for min_pt in range(100, 1000, 100)]

    sns.palplot(sns.cubehelix_palette())

    ax = sns.heatmap(
        auc_mat.transpose(),
        annot=True,
        ax=ax,
        cmap="coolwarm", vmax=0.9, vmin=0.7,
        xticklabels=labels, yticklabels=labels,
        fmt=".3f")

    ax.set_title("AUC of BDT", fontsize=24)
    ax.set_ylabel("Test set", fontsize=18)
    ax.set_xlabel("Training set", fontsize=18)
    ax.invert_yaxis()

    fig.savefig(path_fmt.format(ext="png"))
    fig.savefig(path_fmt.format(ext="pdf"), format="pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, type=str)
    parser.add_argument("--dataset_path", default="./Data", type=str)
    parser.add_argument("--vmax", default=0.9, type=float)
    parser.add_argument("--vmin", default=0.7, type=float)
    args = parser.parse_args()


    main(log_dir=args.log_dir,
         dataset_dir=args.dataset_path,
         out_dir=args.log_dir)
