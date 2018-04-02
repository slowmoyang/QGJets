from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.utils import conv_utils
import keras.backend as K

def get_channel_axis():
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    return channel_axis

def get_channels(x):
    channel_axis = get_channel_axis()
    num_channels = K.int_shape(x)[channel_axis]
    return num_channels

def conv_unit(x,
              filters,
              kernel_size,
              activation="relu",
              order=["conv", "bn", "activation"],
              **kargs):
    if isinstance(order, str):
        order = order.lower().replace(" ", "_")
        if order in ["full_pre-activation"]:
            order = ["bn", "activation", "conv"]
        else:
            raise ValueError("")


    channel_axis = get_channel_axis()

    conv_keys = ["strides", "padding", "dilation_rate",
                 "use_bias", "kernel_initializer", "bias_initializer",
                 "kernel_regularizer", "bias_regularizer",
                 "activity_regularizer", "kernel_constraint",
                 "bias_constraint"]
        
    bn_keys = ["momentum", "epsilon", "center", "scale", "beta_initializer",
               "gamma_initializer", "moving_mean_initializer",
               "moving_variance_initializer", "beta_regularizer",
               "gamma_regularizer", "beta_constraint", "gamma_constraint"]

    conv_kargs = {}
    bn_kargs = {}
    activation_kargs ={}

    if kargs.has_key("name"):
        name = kargs.pop("name")
        conv_kargs["name"] = name + "_Conv2D"
        bn_kargs["name"] = name + "_BatchNorm"
        activation_kargs["name"] = name + "_Activation"

    for key in kargs.keys():
        if key in conv_keys:
            conv_kargs[key] = kargs[key]
        elif key in bn_keys:
            bn_kargs[key] = kargs[key]
        else:
            raise ValueError(":p") 

    if not conv_kargs.has_key("padding"):
        conv_kargs["padding"] = "same"

    activation = "relu" if not kargs.has_key("activation") else kargs["activation"]

    node_dict = {
        "conv": Conv2D(filters, kernel_size, **conv_kargs),
        "bn": BatchNormalization(axis=channel_axis, **bn_kargs),
        "activation": Activation(activation=activation, **activation_kargs)
    }
    
    for each in order:
        x = node_dict[each](x)

    return x


def attach_shortcut(residual_fn):
    def _residual_unit(x):
        F = residual_fn(x)
        y = Add()([x, F])
        return y
    return _residual_fn
