import os
import numpy as np
import pandas as pd

def parse_roc_dpath(dpath):
    dname = dpath.split("/")[-2]
    elem = dname.split("_")
    channels = elem[0]
    kernel = elem[1].split("-")[0][1:]
    return [channels, kernel]



def parse_roc_filename(filename):
    filename = os.path.splitext(filename)[0]
    parsed = filename.split("_")
    step = parsed[2].split("-")[-1]
    auc = parsed[3].split("-")[-1]
    step = int(step) if step.isdigit() else -1
    auc = float(auc)
    return [step, auc]



if __name__ == "__main__":
    logs = "./logs"

    log_dirs = os.listdir(logs)
    roc_dirs = map(
        lambda each: os.path.join(logs, each, "roc"),
        log_dirs)

    best = []
    for each in roc_dirs:
        setup = parse_roc_dpath(each)

        entries = os.listdir(each)
        if len(entries) == 0:
            continue

        dijet = filter(lambda entry: "dijet" in entry, entries)
        zjet = filter(lambda entry: "zjet" in entry, entries)

        dijet = map(parse_roc_filename, dijet)
        zjet = map(parse_roc_filename, zjet)

        dijet = sorted(dijet, key=lambda each: each[1])
        zjet = sorted(zjet, key=lambda each: each[1])
        dijet_best = dijet[-1]
        zjet_best = zjet[-1]
        best.append(setup + dijet_best + zjet_best)

    best = np.array(best)
    df = pd.DataFrame(best)

    df.to_csv("best_auc.csv",  header=["channels", "kernel", "dijet_step", "dijet_auc", "zjet_step", "zjet_auc"])
