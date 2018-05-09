from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd

from keras4jet.utils import get_log_dir
from keras4jet.utils import Directory
from keras4jet.utils import get_filename
from keras4jet.utils import convert_str_to_number



def parse_model_path(path):
    filename = get_filename(path)
    name, step, loss, acc = filename.split("_")
    metadata = [each.split("-") for each in [step, loss, acc]]
    metadata = {key: convert_str_to_number(value) for key, value in metadata}
    metadata.update({"path": path})
    return metadata


def find_good_models(log_dir,
                     lower_better=["loss"],
                     higher_better=["acc"],
                     parsing_fn=parse_model_path):
    """
    min_objs
    
    """
    if isinstance(log_dir, str):
        log_dir = get_log_dir(log_dir, creation=False)
    if isinstance(lower_better, str):
        lower_better = [lower_better, ]
    if isinstance(higher_better, str):
        higher_better = [higher_better, ]
   
    saved_models = log_dir.saved_models.get_entries()
    metadata = [parsing_fn(each) for each in saved_models if not "final" in each]
    metadata = pd.DataFrame(metadata)

    good = []
    for each in lower_better:
        good += list(metadata[metadata[each] == metadata[each].min()]["path"].values)
    for each in higher_better:
        good += list(metadata[metadata[each] == metadata[each].max()]["path"].values)
    good = list(set(good))
    return good
