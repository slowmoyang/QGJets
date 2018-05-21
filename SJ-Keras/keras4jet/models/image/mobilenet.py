from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

from keras import backend as K

from keras.models import Model

from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import Reshape

from keras.applications.mobilenet import DepthwiseConv2D
from keras.applications.mobilenet import _conv_block
from keras.applications.mobilenet import _depthwise_conv_block
from keras.applications.mobilenet import relu6

import numpy as np
import tensorflow as tf

def build_a_model(input_shape,
                  num_classes=2,
                  filters_list=[32, 64, 128, 256],
                  width_multiplier=0.25, # alpha
                  depth_multiplier=1,
                  dropout=1e-3,
                  activation="relu",
                  padding="valid"):


    inputs = Input(input_shape)

    x = _conv_block(inputs, filters_list[0], width_multiplier, strides=(2, 2)) 

    in_channels = filters_list[0]
    for idx, filters in enumerate(filters_list[1:], 1):
        strides = (2, 2) if in_channels < filters else (1, 1)

        x = _depthwise_conv_block(
            inputs=x,
            pointwise_conv_filters=filters,
            alpha=width_multiplier,
            depth_multiplier=depth_multiplier,
            strides=strides,
            block_id=idx)

        in_channels = filters

    # top
    x = GlobalAveragePooling2D()(x)

    shape = (int(filters_list[-1] * width_multiplier), 1, 1)
    x = Reshape(shape)(x)

    x = Dropout(dropout)(x)
    x = Conv2D(num_classes, (1, 1), padding="SAME")(x)
    x = Activation("softmax")(x)
    x = Reshape((num_classes,))(x)

    model = Model(inputs=inputs, outputs=x)    
    return model

def get_custom_objects():
    custom_objects = {}
    custom_objects["DepthwiseConv2D"] = DepthwiseConv2D
    custom_objects["relu6"] = relu6
    return custom_objects
