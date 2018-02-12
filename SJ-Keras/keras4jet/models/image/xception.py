from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import backend as K

from keras.models import Model

from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D

import numpy as np
import tensorflow as tf


from keras4jet.layers import layer_utils
from keras4jet.layers.layer_utils import conv_unit
from keras4jet.layers.xception import separable_conv_unit
from keras4jet.layers.xception import reduction_block
from keras4jet.layers.xception import middle_block

def build_a_model(input_shape,
                  num_classes=2,
                  num_middle_blocks=8,
                  entry_filters=[32, 64, 128, 256, 728],
                  exit_filters=[1024, 1536, 2048]):

    if isinstance(entry_filters, int):
        entry_filters = (entry_filters, )
    if isinstance(exit_filters, int):
        entry_filters = (exit_filters, )


    channel_axis = layer_utils.get_channel_axis()

    # (B, C, 33, 33)
    inputs = Input(input_shape)

    #####################################
    # Entry flow 
    #######################################
    x = conv_unit(inputs, filters=entry_filters[0], kernel_size=3, strides=2, use_bias=False,
                  order=["conv", "bn", "activation"])

    if len(entry_filters) > 1:
        x = conv_unit(x, filters=entry_filters[1], kernel_size=3, strides=1, use_bias=False,
                      order=["conv", "bn", "activation"])

    if len(entry_filters) > 2:
        for filters in entry_filters[2:]:
            x = reduction_block(x, filters)

    ################################################
    #    Middle flow
    #######################################################
    in_channels = K.int_shape(x)[channel_axis]
    for _ in range(num_middle_blocks):
        x = middle_block(x, filters=in_channels)

    ####################################################
    #    Exit flow
    ###########################################
    x = reduction_block(x, filters=[in_channels, exit_filters[0]])
    if len(exit_filters) > 1:
        for filters in exit_filters[1:]:
            x = separable_conv_unit(x, filters=filters)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=num_classes)(x)
    x = Activation("softmax")(x)

    model = Model(inputs=inputs, outputs=x)
    return model

