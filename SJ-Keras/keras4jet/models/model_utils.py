from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import BatchNormalization 
from keras.layers import GlobalAveragePooling2D

import tensorflow as tf

_ACT_NAME_SET = {"act", "activation"}
_BN_NAME_SET = {"bn", "batch_norm", "batch_normalization"}
_CONV2D_NAME_SET = {"conv2d", "convolution2d", "conv", "convolution"}
_DENSE_NAME_SET = {"dense"}
_DROPOUT_NAME_SET = {"dropout"}


def get_channel_axis():
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    return channel_axis


def dense_block(x,
                units,
                order=["dense", "act", "bn"],
                activation="relu",
                rate=0.5):

    layer_order = [each.lower().replace(" ", "_") for each in layer_order]
    if use_bias is None:
        has_bn = bool(_BN_NAME_SET.intersection(order))
        use_bias = False if has_bn else True
        
    for layer in order:
        if layer in _DENSE_NAME_SET:
            x = Dense(units, activation=None, use_bias=use_bias)(x)
        elif layer in _ACT_NAME_SET:
            x = Activation(activation)
        elif layer in _BN_NAME_SET:
            x = BatchNormalization(axis=-1)(x)
        elif layer in _DROPOUT_NAME_SET:
            x = Dropout(rate=rate)(x)
        else:
            raise ValueError

def conv_block(x,
               filters,
               kernel_size,
               strides,
               layer_order=["bn", "act", "conv"],
               padding="VALID",
               use_bias=None,
               activation="relu"):
    layer_order = [each.lower().replace(" ", "_") for each in layer_order]
    if use_bias is None:
        has_bn = bool(_BN_NAME_SET.intersection(layer_order))
        use_bias = False if has_bn else True

    for layer in layer_order:
        if layer in _CONV2D_NAME_SET:
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
        elif layer in _ACT_NAME_SET:
            x = Activation(activation)(x)
        elif layer in _BN_NAME_SET:
            channel_axis = get_channel_axis()
            x = BatchNormalization(axis=channel_axis)(x)
        else:
            raise ValueError
    return x

