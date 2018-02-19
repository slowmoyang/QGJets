"""
arXiv:1709.01507 [cs.CV]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K

from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Multiply
from keras.layers import Reshape

from keras4jet.layers import layer_utils

def se_block(x, reduction_ratio=4, activation="relu"):
    channel_axis = layer_utils.get_channel_axis()
    in_channels = K.int_shape(x)[channel_axis]
    bottleneck_units = int(in_channels / reduction_ratio)

    # Squeeze
    z = GlobalAveragePooling2D()(x) # [C, ]

    # Excitation
    s = Dense(bottleneck_units, activation="relu")(z)
    s = Dense(units=in_channels, activation="relu")(s)
    s = Reshape((in_channels, 1, 1))(s)

    x_tilde = Multiply()([s, x])
    return x_tilde
