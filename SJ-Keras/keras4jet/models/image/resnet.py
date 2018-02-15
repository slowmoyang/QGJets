from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import keras.backend as K
import tensorflow as tf

from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D 

from keras.models import Model

from keras4jet.layers.resnet import residual_unit


def build_a_model(input_shape,
                  filters_list,
                  bottleneck=True,
                  num_classes=2):

    input_image = Input(input_shape)

    x = Conv2D(32, kernel_size=5, strides=2, padding="SAME")(input_image)

    for filters in filters_list:
        x = residual_unit(x, filters, bottleneck=bottleneck)

    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    logits = Dense(units=num_classes)(x)
    outputs = Activation('sigmoid')(logits)

    model = Model(inputs=input_image, outputs=outputs)

    return model
    
def get_custom_objects():
    custom_objects = {}
    custom_objects["tf"] = tf
    return custom_objects
