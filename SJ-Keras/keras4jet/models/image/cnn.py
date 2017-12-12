from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import keras.backend as K
import tensorflow as tf

from keras.layers import Input
from keras.layers import (
    Conv2D,
    GlobalAveragePooling2D,
    Activation,
    Dense,
    Flatten
)
from keras.models import Model, Sequential


def build_a_model(input_shape, num_classes=1):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
    
