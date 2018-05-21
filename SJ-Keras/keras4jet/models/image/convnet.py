from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from ..model_utils import get_channel_axis
from ..model_utils import conv_block
from ..model_utils import dense_block

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization 
from keras.layers import GlobalAveragePooling2D

import numpy as np
import tensorflow as tf


def build_a_model(input_shape,
                  num_classes=2,
                  kernel_size=11,
                  filters_list=[16, 32, 32, 64, 64],
                  last_hidden_units=None,
                  activation="relu",
                  padding="VALID",
                  top="proj_gap"):
    inputs = Input(input_shape)

    hidden = Conv2D(filters=filters_list[0], kernel_size=kernel_size, strides=2, padding=padding)(inputs)
    hidden = Activation(activation)(hidden)

    for i, filters in enumerate(filters_list[1:]):
        hidden = conv_block(
            x=hidden,
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            activation=activation)
        #if (i != 0) and (i % 2 == 0):
        #    hidden = MaxPooling2D(2)(hidden)

    if top == "proj_gap":
        hidden = Conv2D(filters=num_classes, kernel_size=1, strides=1, padding="SAME")(hidden)
        logits = GlobalAveragePooling2D()(hidden)
    elif top == "gap_dense":
        hidden = GlobalAveragePooling2D()(hidden)
        if last_hidden_units is not None:
            hidden = dense_block(last_hidden_units, activation=activation)
        logits = Dense(num_classes)(hidden)
    elif top == "flatten_dense":
        hidden = Flatten()(hidden)
        if last_hidden_units is not None:
            hidden = dense_block(last_hidden_units, activation=activation)
        logits = Dense(num_classes)(hidden)

    outputs = Activation("softmax")(logits)

    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = build_a_model((1,33,33), 2)
