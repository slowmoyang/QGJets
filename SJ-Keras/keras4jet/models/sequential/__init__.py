from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import ak4_without_res
from . import ak4_elu
from . import ak4_elu_gru
from . import ak4_without_conv

__all__ = ["ak4"]

def build_a_model(model_name, *args, **kargs):
    return getattr(models, model_name).build_a_model(*args, **kargs)
