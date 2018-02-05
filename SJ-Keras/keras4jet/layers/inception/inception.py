"""
ref. arXiv:1409.4842 [cs.CV]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.utils import conv_utils
import keras.backend as K

from keras4jet.layers import layer_utils
from keras4jet.layers.layer_utils import conv_unit


def factorized_conv(x, filters, kernel_size, factorization, strides=(1,1), **kargs):
    kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    strides = conv_utils.normalize_tuple(strides, 2, 'strides')

    k = kernel_size[0]
    assert bool(k%2)

    factorization = factorization.upper()
    if factorization == "A":
        num_iter = int( (k - 1) / 2)
        for _ in range(num_iter):
            x = conv_unit(x, filters, kernel_size=(3,3), strides=strides, **kargs)
    elif factorization == "B":
        x = conv_unit(x, filters, kernel_size=(1, k), strides=(1, strides[1]), **kargs)
        x = conv_unit(x, filters, kernel_size=(k, 1), strides=(strides[0], 1), **kargs)
    elif factorization == "C":
        x0 = conv_unit(x, filters, kernel_size=(1, k), strides=(1, stirdes[1]), **kargs)
        x1 = conv_unit(x, filters, kernel_size=(k, 1), strides=(strides[0], 1), **kargs)
        x = Concatenate(axis=channel_axis)([x_height, x_width])
    else:
        raise ValueError(":p")

    return x


def pool_proj(x, filters, pooling=AveragePooling2D):
    x = pooling(pool_size=(3,3), strides=(1,1), padding="SAME")(x)
    x = conv_unit(x, filters=filters, kernel_size=(1,1))
    return x



def inception_a(x, filters=None):
    """
    For 35 X 35 grid modules of the pure Inception-v4 network.
    This is the Inception-A block.

    In: [384, 35, 35]
    Out: [384, 35, 35]
    """
    channel_axis = layer_utils.get_channel_axis()

    if filters is None:
        in_channels = K.int_shape(x)[channel_axis]
        unit_filters = int(in_channels/4)
        filters = {
            "branch0": unit_filters,
            "branch1": unit_filters,
            "branch2_0": int(2 / 3 * unit_filters),
            "branch2_1": unit_filters,
            "branch3_0": int(2 / 3 * unit_filters),
            "branch3_1": unit_filters,
            "branch3_2": unit_filters
    }

    # Avg Pooling --> 1x1 Conv
    x0 = pool_proj(x, filters=filters["branch0"])

    # 1x1 Conv (96)
    x1 = conv_unit(x, kernel_size=1, filters=filters["branch1"])

    # 1x1 Conv (64) --> 3x3 Conv (96)
    x2 = conv_unit(x, kernel_size=1, filters=filters["branch2_0"])
    x2 = conv_unit(x2, kernel_size=3, filters=filters["branch2_1"])

    # 1x1 Conv (64) --> 
    x3 = conv_unit(x, kernel_size=1, filters=filters["branch3_0"])
    x3 = conv_unit(x3, kernel_size=3, filters=filters["branch3_1"])
    x3 = conv_unit(x3, kernel_size=3, filters=filters["branch3_2"])

    # Filter concat
    filter_concat = Concatenate(axis=channel_axis)([x0, x1, x2, x3])
    return filter_concat


def inception_b(x, filters=None):
    """
    For 17 X 17 grid modules of the pure Inception-v4 network.
    This is the Inception-B block.

    In: [1024, 17, 17]
    Out: [1024, 17, 17]
    """
    channel_axis = layer_utils.get_channel_axis()

    if filters is None:
        in_channels = K.int_shape(x)[channel_axis]
        filters = {
            "branch0": int(in_channels/8), # 1 / 8
            "branch1": int(0.375 * in_channels), # 3/8

            "branch2_0": int(0.1875 * in_channels), # 3 / 16
            "branch2_1": int(0.21875 * in_channels), # 7 / 32
            "branch2_2": int(0.25 * in_channels), # 1 / 4

            "branch3_0": int(0.1875 * in_channels),
            "branch3_1": int(0.1875 * in_channels),
            "branch3_2": int(0.21875 * in_channels),
            "branch3_3": int(0.21875 * in_channels), 
            "branch3_4": int(0.25 * in_channels)
    }



    x0 = pool_proj(x, filters=filters["branch0"])

    x1 = conv_unit(x, kernel_size=(1,1), filters=filters["branch1"])

    x2 = conv_unit(x, kernel_size=(1, 1), filters=filters["branch2_0"])
    x2 = conv_unit(x2, kernel_size=(1, 7), filters=filters["branch2_1"])
    x2 = conv_unit(x2, kernel_size=(7, 1), filters=filters["branch2_2"])

    x3 = conv_unit(x, kernel_size=(1, 1), filters=filters["branch3_0"])
    x3 = conv_unit(x3, kernel_size=(1, 7), filters=filters["branch3_1"])
    x3 = conv_unit(x3, kernel_size=(7, 1), filters=filters["branch3_2"])
    x3 = conv_unit(x3, kernel_size=(1, 7), filters=filters["branch3_3"])
    x3 = conv_unit(x3, kernel_size=(7, 1), filters=filters["branch3_4"])


    # Filter concat
    filter_concat = Concatenate(axis=channel_axis)([x0, x1, x2, x3])
    return filter_concat

def inception_c(x, filters=None):
    """
    For 8 X 8 grid modules of the pure Inception-v4 network.
    This is the Inception-C block.

    In: [1536, 8, 8]
    Out: [1536, 8, 8]
    """
    channel_axis = layer_utils.get_channel_axis()

    if filters is None:
        in_channels = K.int_shape(x)[channel_axis]
        filters = {
            "branch0": int(in_channels / 8),

            "branch1": int(in_channels / 6),

            "branch2_0": int(in_channels / 4),
            "branch2_1": int(in_channels / 6),
            "branch2_2": int(in_channels / 6),

            "branch3_0": int(in_channels / 4),
            "branch3_1": int( ( 7 / 24 ) * in_channels),
            "branch3_2": int(in_channels / 3),
            "branch3_3": int(in_channels / 6), 
            "branch3_4": int(in_channels / 6)
    }


    x0 = pool_proj(x, filters=filters["branch0"])

    x1 = conv_unit(x, kernel_size=(1,1), filters=filters["branch1"])

    x2 = conv_unit(x, kernel_size=(1, 1), filters=filters["branch2_0"])
    x2_1x3 = conv_unit(x2, kernel_size=(1, 3), filters=filters["branch2_1"])
    x2_3x1 = conv_unit(x2, kernel_size=(3, 1), filters=filters["branch2_2"])

    x3 = conv_unit(x, kernel_size=(1, 1), filters=filters["branch3_0"])
    x3 = conv_unit(x3, kernel_size=(1, 3), filters=filters["branch3_1"])
    x3 = conv_unit(x3, kernel_size=(3, 1), filters=filters["branch3_2"])
    x3_3x1 = conv_unit(x3, kernel_size=(3, 1), filters=filters["branch3_3"])
    x3_1x3 = conv_unit(x3, kernel_size=(1, 3), filters=filters["branch3_4"])


    # Filter concat
    filter_bank = [x0, x1, x2_1x3, x2_3x1, x3_3x1, x3_1x3]
    filter_concat = Concatenate(axis=channel_axis)(filter_bank)
    return filter_concat


if __name__ == "__main__":
    from keras.layers import Input

    x = Input((256, 33, 33))

    inc_a = inception_a(x)
    inc_b = inception_b(inc_a)
    inc_c = inception_c(inc_b)
