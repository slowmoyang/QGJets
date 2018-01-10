from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import (
    Input,
    Activation,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from keras.engine.topology import Layer

from keras.utils import multi_gpu_model
from keras import backend as K




def build_a_model(input_shape, num_classes=1):
    def bn_act_conv(x, filters, act_fn):
        out = BatchNormalization(axis=1)(x)
        out = Activation(act_fn)(out)
        out = Conv2D(filters, (3,3), padding="SAME")(out)
        return out

    inputs = Input(input_shape)

    out = bn_act_conv(inputs, 32, "elu")
    out = bn_act_conv(out, 32, "elu")
    out = MaxPooling2D((2,2))(out)

    out = bn_act_conv(out, 64, "elu")
    out = bn_act_conv(out, 64, "elu")
    out = bn_act_conv(out, 64, "elu")
    out = bn_act_conv(out, 64, "elu")
    out = MaxPooling2D((2,2))(out)

    out = bn_act_conv(out, 128, "elu")
    out = bn_act_conv(out, 128, "elu")
    out = bn_act_conv(out, 128, "elu")
    out = bn_act_conv(out, 128, "elu")
    out = MaxPooling2D((2,2))(out)

    # 
    out = Conv2D(filters=num_classes, kernel_size=(1, 1))(out)
    out = GlobalAveragePooling2D()(out)
    out = Activation("sigmoid")(out)

    model = Model(inputs=inputs, outputs=out)
    return model
