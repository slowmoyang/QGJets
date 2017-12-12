from __future__ import absolute_import

import keras4jet.models.image
import keras4jet.models.sequential

__all__ = []

def build_a_model(model_name, *args, **kargs):
    try:
        return getattr(keras4jet.models.image, model_name).build_a_model(*args, **kargs)
    except AttributeError:
        return getattr(keras4jet.models.sequential, model_name).build_a_model(*args, **kargs)
