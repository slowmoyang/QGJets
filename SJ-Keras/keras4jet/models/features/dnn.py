from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

from keras import backend as K

from keras.models import Model

from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization 
from keras.layers import Dense
from keras.layers import Dropout

import numpy as np
import tensorflow as tf


def block(x, units, activation="relu", rate=0.5):
    x = Dense(units, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(rate=rate)(x)
    return x


def build_a_model(num_features,
                  units_list,
                  num_classes=2,
                  activation="relu",
                  rate=0.5):

    inputs = Input((num_features, ))

    x = inputs
    for units in units_list:
        x = block(x, units, activation=activation, rate=rate)

    logits = Dense(units=num_classes)(x)
    outputs = Activation("softmax")(logits)

    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = build_a_model((1,33,33), 2)
