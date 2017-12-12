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
    Activation
)
from keras.models import Model

from keras4jet.layers.resnet import residual_unit


def build_a_model(input_shape, num_classes=1):

    input_image = Input(input_shape)

    x = Conv2D(32, kernel_size=5, strides=2, padding="SAME")(input_image)

    filters_list = [128, 128, 256, 256, 256, 512, 512]
    for filters in filters_list:
        x = residual_unit(x, filters, bottleneck=True)

    x = Conv2D(filters=num_classes, kernel_size=1, strides=1)(x)
    logits = GlobalAveragePooling2D()(x)
    sigmoid = Activation('sigmoid')(logits)

    model = Model(inputs=input_image, outputs=sigmoid)

    return model
    
