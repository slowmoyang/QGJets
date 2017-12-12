from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import keras.backend as K
import tensorflow as tf

from keras import layers
from keras.layers import (
    Input,
    Conv2D,
    Activation,
    BatchNormalization,
    GlobalAveragePooling2D
)

from keras.models import Model

from keras4jet.models.image.resnet import zero_padding_shortcut

def asym_conv(x, filters, k=3, s=1):
    x = Conv2D(
        filters=filters, kernel_size=(1, k),
        strides=(1, s), padding="SAME")(x)
    x = Conv2D(
        filters=filters, kernel_size=(k, 1),
        strides=(s, 1), padding="SAME")(x)
    return x


# Bottleneck and full pre-activation
def bottleneck_residual_function(x, filters):
    input_channels = x.shape[1].value
    s = 2 if input_channels < filters else 1 # strides

    bottleneck_filters = int(filters/4)

    #
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = asym_conv(x, filters=bottleneck_filters, s=s)
    #
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = asym_conv(x, filters=bottleneck_filters)
    #
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = asym_conv(x, filters=filters)
    #
    return x


def residual_unit(x, filters):
    input_channels = x.shape[1].value
    if input_channels < filters:
        h = zero_padding_shortcut(x, filters)
    else:
        h = x
    F = bottleneck_residual_function(x, filters)
    y = layers.add([F, h])
    return y 


def build_a_model(input_shape, filters_list, num_classes=1):
    input_image = Input(input_shape)

    x = asym_conv(input_image, filters=8, k=5, s=2)
    for filters in filters_list:
        x = residual_unit(x, filters)

    x = Conv2D(filters=num_classes, kernel_size=1, strides=1)(x)
    logits = GlobalAveragePooling2D()(x) 
    sigmoid = Activation('sigmoid')(logits)

    model = Model(inputs=input_image, outputs=sigmoid)
    return model
