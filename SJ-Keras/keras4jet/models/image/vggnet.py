from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

from keras import backend as K

from keras.models import Model

from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization 
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import MaxPooling2D

import numpy as np
import tensorflow as tf

from keras4jet.layers.layer_utils import conv_unit

def build_a_model(input_shape,
                  num_classes=2,
                  num_layers=[2, 2, 3, 3, 3],
                  init_filters=32,
                  dense_units=256,
                  order=["bn", "activation", "conv"],
                  activation="relu"):

    inputs = Input(input_shape)

    x = inputs

    filters = init_filters
    for i, num_conv in enumerate(num_layers):
        for _ in range(num_conv):
            x = conv_unit(x, filters=filters, kernel_size=(3, 3), activation=activation, order=order)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        filters *= 2

    x = Flatten()(x)
    x = Dense(units=dense_units)(x)
    x = Activation(activation)(x)
    x = Dense(units=dense_units)(x)
    x = Activation(activation)(x)

    logits = Dense(units=num_classes)(x)
    y_pred = Activation("softmax")(logits)

    model = Model(inputs=inputs, outputs=y_pred)
    return model

if __name__ == "__main__":
    model = build_a_model((1,33,33), 2)
