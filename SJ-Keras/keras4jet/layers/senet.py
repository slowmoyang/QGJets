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



if __name__ == "__main__":
    import layer_utils
    from layer_utils import conv_unit
    from layer_utils import factorized_conv
else:
    from . import layer_utils
    from .layer_utils import factorized_conv


def se_block(x, reduction_ratio=4, activation="relu"):
    channel_axis = layer_utils.get_channel_axis()
    in_channels = K.int_shape(x)[channel_axis]
    bottleneck_units = int(in_channels / reduction_ratio)

    # Squeeze
    z = GlobalAveragePooling2D()(x)

    # Excitation
    s = Dense(bottleneck_units)(z)
    s = Activation(activation)(s)
    s = Dense(channels, name=)(s)
    s = Activation("sigmoid")(s)
    s = Reshape()(s)

    x_tilde = s * x
    return x_tilde
