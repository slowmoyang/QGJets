from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from datetime import datetime
import copy
from collections import OrderedDict
import argparse
import time
import json

import xgboost as xgb

import numpy as np
from sklearn import metrics

import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

import ROOT
from ROOT import TFile
from ROOT import TH1F
from ROOT import TCanvas
from ROOT import gDirectory
from ROOT import gStyle
from ROOT import gROOT
gROOT.SetBatch(True)

from data_loading import load_dataset
from utils import Directory
from utils import Config
from utils import parse_dataset_path



def main():
    ###############################
    # Argument Parsing
    ##############################
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_names", nargs="+", type=str,
        default=["ptD", "axis1", "axis2", "cmult", "nmult"])

    parser.add_argument("--extra", nargs="+", type=str,
        default=["pt", "eta"])

    parser.add_argument("--log_dname", default="XGB", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--test_path", type=str)

    # General Parameters
    parser.add_argument("--booster", default="gbtree", type=str)
    parser.add_argument("--silent", default=False, type=bool)

    # Parameters for Tree Booster
    parser.add_argument("--learning_rate", default=0.3, type=float)
    parser.add_argument("--max_depth", default=4, type=int,
        help="maximum depth of a tree, increase this value will make the model \
              more complex / likely to be overfitting. 0 indicates no limit, \
              limit is required for depth-wise grow policy.")
    parser.add_argument("--n_estimators", default=100, type=int)
    parser.add_argument("--min_split_loss", default=0, type=float,
        help="minimum loss reduction required to make a further partition on \
              a leaf node of the tree.")
    parser.add_argument("--lambda", default=1, type=float,
        help="L2 regularization term on weights")
    parser.add_argument("--alpha", default=0, type=float,
        help="L1 regularization term on weights")

    # Additional parameters for Dart Booster
    parser.add_argument("--sample_type", default="uniform", help='"uniform" or "weighted"')
    parser.add_argument("--normalize_type", default="tree", help='"tree" or "forest"')
    parser.add_argument("--rate_drop", default=0.1, help="range: [0.0, 1.0]")
    parser.add_argument("--one_drop", default=0)
    parser.add_argument("--skip_drop", default=0.5, help="range: [0.0, 1.0]")

    # XGBClassifier.fit parameters
    parser.add_argument("--early_stopping_rounds", default=5,
        help="Activates early stopping. Validation error needs to decrease at \
              least every <early_stopping_rounds> round(s) to continue training.")

    args = parser.parse_args()

    ############################################
    # Log Directory
    ########################################
    min_pt, max_pt = parse_dataset_path(args.train_path)

    log_dname = "{}_{}_{}".format("root", min_pt, max_pt)

    log_dpath = os.path.join("./logs", log_dname)
    log_dir = Directory(log_dpath, creation=True)

    ################################3
    # Config
    #############################
    config = Config(log_dir.path, mode="write")
    config.update(args)

    ########################
    # Data Loading
    #######################
    #train_path, valid_path, test_path = get_path_of_datasets(
    #    event_type=config.event_type, min_pt=config.min_pt, max_pt=config.max_pt)


    config["training_path"] = args.train_path
    config["validation_path"] = args.valid_path
    config["test_path"] = args.test_path

    train_set, train_extra = load_dataset(args.train_path, features=config.feature_names, extra=config.extra) 
    valid_set, valid_extra = load_dataset(args.valid_path, features=config.feature_names, extra=config.extra) 
    test_set, test_extra = load_dataset(args.test_path, features=config.feature_names, extra=config.extra) 

    config["num_train_set"] = train_set[0].shape[0]
    config["num_valid_set"] = valid_set[0].shape[0]
    config["num_test_set"] = test_set[0].shape[0]

    ##############################################
    # Parameters
    ##################################################
    params = dict()
    # General Parameters
    params["booster"] = args.booster
    params["silent"] = args.silent
    # Learning Task Parameters
    params["objective"] = "binary:logistic"
    params["base_score"] = 0.5
    params["eval_metric"] = ["error", "auc", "logloss"]
    params["seed"] = 0
    # Parameters for Tree Booster
    params["learning_rate"] = args.learning_rate
    params["max_depth"] = args.max_depth
    params["n_estimators"] = args.n_estimators
    params["min_split_loss"] = args.min_split_loss
    # Additional parameters for Dart Booster
    if args.booster == "dart":
        params["sample_type"] = args.sample_type
        params["normalize_type"] = args.normalize_type
        params["rate_drop"] = args.rate_drop
        params["one_drop"] = args.one_drop
        params["skip_drop"] = args.skip_drop
    # Parameters for GPU 
    params['gpu_id'] = 0 
    params['max_bin'] = 16
    params['tree_method'] = 'gpu_hist'
    # params["n_gpu"] = -1
    # params["predictor"] = "gpu_predictor"

    config.update(params)


    #########################################
    # Classifier
    ########################################
    clf = xgb.XGBClassifier(**params)

    start = time.time()

    clf.fit(
        X=train_set[0],
        y=train_set[1],
        eval_set=[train_set, valid_set],
        early_stopping_rounds=config["early_stopping_rounds"],
        verbose=True)

    duration = time.time() - start
    print("Training time: {:.1f} sec".format(duration))

    fname = os.path.join(log_dir.path, "xgb.model")
    clf._Booster.save_model(fname)
    config["model"] = fname

    ###########################
    #   Evaluation
    ##########################
    x, y_true, _ = test_set
    y_score = clf.predict_proba(x)[:, 1]
    y_pred = clf.predict(x)

    # Accuracy
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    config["accuracy"] = acc
    print("Accuracy: {:.3f}".format(acc))


    # Training/Validation Metric Curves
    results = clf.evals_result() 
    num_epochs = len(results["validation_0"]["error"])
    x_axis = range(num_epochs)
    path_fmt = os.path.join(log_dir.path, "{fname}.{ext}")
    for metric in config["eval_metric"]:
        fig, ax = plt.subplots()
        ax.plot(x_axis, results["validation_0"][metric], label="Training")
        ax.plot(x_axis, results["validation_1"][metric], label="Validation")
        ax.legend()
        ax.set_ylabel(metric)
        ax.set_xlabel("# of epochs")
        fig.savefig(path_fmt.format(fname=metric, ext="png"))
        fig.savefig(path_fmt.format(fname=metric, ext="pdf"), format="pdf")

    # ROC Cruve
    fpr, tpr, _ = metrics.roc_curve(y_true=y_true, y_score=y_score)
    fnr = 1 - fpr

    auc = metrics.auc(x=tpr, y=fnr)
    config["auc"] = auc
    print("AUC: {:.2f}".format(auc))

    roc_path_fmt = os.path.join(log_dir.path, "roc_curve_auc_{}.{}".format(auc, "{ext}"))

    np.savez(roc_path_fmt.format(ext="npz"), tpr=tpr, fnr=fnr)


    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(tpr, fnr, color='darkorange', lw=2,
             label='ROC curve (auc = {:.3f})'.format(auc))
    plt.plot([0,1], [1,1], color='navy', lw=2, linestyle='--')
    plt.plot([1,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.1])
    plt.ylim([0.0, 1.1])
    plt.xlabel('Quark Jet Efficiency (TPR)')
    plt.ylabel('Gluon Jet Rejection (FNR)')
    plt.title('BDT')
    plt.legend(loc="lower left")
    ax.grid()

    fig.savefig(roc_path_fmt.format(ext="png"))
    fig.savefig(roc_path_fmt.format(ext="pdf"), format="pdf")


    # Output histogram
    out_path_fmt = os.path.join(log_dir.path, "out_hist.{ext}")
    out_file = TFile.Open(out_path_fmt.format(ext="root"), "RECREATE")

    out_file.mkdir("train")
    out_file.cd("train")

    train_quark = TH1F("quark", "quark", 100, 0, 1)
    train_gluon = TH1F("gluon", "gluon", 100, 0, 1)

    y_train_score = clf.predict_proba(train_set[0])[:, 1]
    for is_gluon, score in zip(train_set[1], y_train_score):
        if is_gluon:
            train_gluon.Fill(score)
        else:
            train_quark.Fill(score)    

    out_file.cd()
    out_file.mkdir("test")
    out_file.cd("test")

    test_quark = TH1F("quark", "quark", 100, 0, 1)
    test_gluon = TH1F("gluon", "gluon", 100, 0, 1)

    for is_gluon, score in zip(y_true, y_score):
        if is_gluon:
            test_gluon.Fill(score)
        else:
            test_quark.Fill(score)  

    out_file.cd()

    out_file.Write()


    can = TCanvas("can", "can", 800, 600)
    keys = ["train/quark", "train/gluon", "test/quark", "test/gluon"]
    hists = [copy.deepcopy(out_file.Get(key)) for key in keys]

    for h in hists:
        h.Scale(1.0 / h.Integral())

    hists[0].Draw('hist')

    for h in hists[1:]:
        h.Draw('hist SAME')

    maximum = max(h.GetMaximum() for h in hists)
    hists[0].SetMaximum(maximum + 0.005)

    hists[0].SetTitle("BDT Output")
        
    colors = [38, 46, 4, 2]

    hists[0].SetFillColorAlpha(colors[0], 0.5)
    hists[1].SetFillColorAlpha(colors[1], 0.5)

    hists[2].SetLineColor(colors[2])
    hists[3].SetLineColor(colors[3])

    hists[2].SetLineWidth(3)
    hists[3].SetLineWidth(3)

    for h, c in zip(hists, colors):
        h.SetLineColor(c)
        
    leg = ROOT.TLegend(0.4, 0.6, 0.6, 0.8)
    for idx, (h, k) in enumerate(zip(hists, keys)):
        option = "f" if idx < 2 else "l"
        leg.AddEntry(h, k, option)

    leg.Draw() 
    gStyle.SetOptStat(0)
    can.SetGrid()

    can.SaveAs(out_path_fmt.format(ext="png"))
    can.SaveAs(out_path_fmt.format(ext="pdf"))

    out_file.Close()

    ###########################################
    # Feature Importance
    ##########################################
    imp_path_fmt = os.path.join(log_dir.path, "feature_importance.{ext}")

    importance = clf.get_booster().get_fscore()
    normalizer = sum(importance.values())

    feature_names = train_set[2]

    name_and_score = []
    for name, score in importance.iteritems():
        idx = int(name.lstrip("f"))
        name_and_score.append((feature_names[idx], score))

    importance = {name: score / normalizer for name, score in name_and_score}

    importance = OrderedDict(sorted(importance.iteritems(), key=lambda item: item[1]))



    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)

    ax.bar(range(len(importance)), list(importance.values()), align="center")
    ax.set_xticks(range(len(importance)))
    ax.set_xticklabels(list(importance.keys()), fontdict={"fontsize": 18})

    ax.set_title("Feature Importance", fontsize=24)
    ax.set_xlabel("Features", fontsize=20)
    ax.set_ylabel("Relative Importance", fontsize=20)

    fig.savefig(imp_path_fmt.format(ext="png"))
    fig.savefig(imp_path_fmt.format(ext="pdf"), format="pdf")

    unused_features = list(set(feature_names).difference(importance.keys()))
    importance.update({key: 0.0 for key in unused_features})

    with open(imp_path_fmt.format(ext="json"), "w") as json_file:
        dumped = json.dumps(importance)
        json_file.write(dumped)



    #################
    # END
    ##############
    config.save()

if __name__ == "__main__":
    main()
