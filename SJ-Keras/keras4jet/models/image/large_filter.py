from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

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

def block(filters, kernel_size, strides, padding, activation):
    def _block(x):
        out = BatchNormalization()(x)
        out = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(out)
        out = Activation(activation)(out)
        return out
    return _block


def build_a_model(input_shape,
                  num_classes=2,
                  kernel_size=11,
                  filters_list=[16, 32, 32, 64, 64],
                  activation="relu",
                  padding="SAME"):
    inputs = Input(input_shape)

    out = Conv2D(filters=filters_list[0], kernel_size=kernel_size, strides=2, padding=padding)(inputs)
    out = Activation(activation)(out)
    for i, filters in enumerate(filters_list[1:]):
        out = block(filters, kernel_size, 1, padding, "relu")(out)
        if (i != 0) and (i % 2 == 0):
            out = MaxPooling2D(2)(out)

    out = Conv2D(filters=num_classes, kernel_size=1, strides=1, padding="SAME")(out)
    out = GlobalAveragePooling2D()(out)
    out = Activation("softmax")(out)

    model = Model(inputs=inputs, outputs=out)
    return model

if __name__ == "__main__":
    model = build_a_model((1,33,33), 2)
