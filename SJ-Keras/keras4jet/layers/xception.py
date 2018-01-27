"""
arXiv:1610.02357 [cs.CV] (https://arxiv.org/abs/1610.02357)
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



def middle_block(x, filters=728, activation="relu", num_iter=3):

    channel_axis = layer_utils.get_channel_axis()

    residual = x

    for _ in range(num_iter):
        x = Activation(activation)(x)
        x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)

    x = Add()([x, residual])



