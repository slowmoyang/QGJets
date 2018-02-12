"""
arXiv:1610.02357 [cs.CV] (https://arxiv.org/abs/1610.02357)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K

from keras.layers import Activation
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D

from keras4jet.layers import layer_utils
from keras4jet.layers.layer_utils import conv_unit


def separable_conv_unit(x,
                        filters,
                        kernel_size=(3,3),
                        order=["activation", "bn", "conv"],
                        **kargs):
    channel_axis = layer_utils.get_channel_axis()

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

    activation = kargs["activation"] if kargs.has_key("activation") else "relu"

    node_dict = {
        "conv": SeparableConv2D(filters, kernel_size, **conv_kargs),
        "bn": BatchNormalization(axis=channel_axis, **bn_kargs),
        "activation": Activation(activation=activation, **activation_kargs)
    }
    
    for each in order:
        x = node_dict[each](x)

    return x


def reduction_block(x, filters):
    if isinstance(filters ,tuple):
        pass
    elif isinstance(filters, int):
        filters = (filters, filters)
    elif isinstance(filters, list):
        filters = tuple(filters)

    residual = Conv2D(filters[1], (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = separable_conv_unit(x, filters[0])
    x = separable_conv_unit(x, filters[1])

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, residual])
    return x


def middle_block(x, filters, num_blocks=3):
    channel_axis = layer_utils.get_channel_axis()
    residual = x
    for _ in range(num_blocks):
        x = separable_conv_unit(x, filters)
    x = Add()([x, residual])
    return x
