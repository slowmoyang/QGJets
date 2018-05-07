from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd

from keras4jet.utils import get_log_dir
from keras4jet.utils import Directory


def get_filename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def convert_str_to_number(string):
    float_case = float(string)
    int_case = int(float_case)
    output = int_case if float_case == int_case else float_case
    return output


def parse_model_path(path):
    filename = get_filename(path)
    name, step, loss, acc, auc = filename.split("_")
    metadata = [each.split("-") for each in [step, loss, acc, auc]]
    metadata = {key: convert_str_to_number(value) for key, value in metadata}
    metadata.update({"path": path})
    return metadata


def find_good_models(log_dir):
    if isinstance(log_dir, str):
        log_dir = get_log_dir(log_dir, creation=False)
    
    saved_models = log_dir.saved_models.get_entries()
    metadata = [parse_model_path(each) for each in saved_models if not "final" in each]
    metadata = pd.DataFrame(metadata)

    good = []
    good += list(metadata[metadata["loss"] == metadata["loss"].min()]["path"].values)
    good += list(metadata[metadata["acc"] == metadata["acc"].max()]["path"].values)
    good += list(metadata[metadata["auc"] == metadata["auc"].max()]["path"].values)
    good = list(set(good))
    return good
