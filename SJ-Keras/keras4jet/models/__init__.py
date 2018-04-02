from __future__ import absolute_import

import keras4jet.models.image
import keras4jet.models.sequential
import keras4jet.models.features

__all__ = []

def build_a_model(model_type, model_name, *args, **kargs):
    models_subdir = getattr(keras4jet.models, model_type)
    return getattr(models_subdir, model_name).build_a_model(*args, **kargs)

def get_custom_objects(model_type, model_name):
    models_subdir = getattr(keras4jet.models, model_type)
    model_file = getattr(models_subdir, model_name)
    if hasattr(model_file, "get_custom_objects"):
        custom_objects = model_file.get_custom_objects()
    else:
        custom_objects = dict()
    return custom_objects
