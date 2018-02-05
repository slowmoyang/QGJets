"""
ref. arXiv:1409.4842 [cs.CV]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from keras.layers import MaxPooling2D
from keras.layers import Concatenate
import keras.backend as K

from keras4jet.layers import layer_utils
from keras4jet.layers.layer_utils import conv_unit


def reduction_a(x,
                filters=None,
                order=["conv", "bn", "activation"],
                **kargs):
    """
    for 35 x 35 to 17 x 17 reduction module.

    In: [384, 35, 35]
    Out: [1024, 17, 17]
    """
    channel_axis = layer_utils.get_channel_axis()
    if filters is None:
        in_channels = K.int_shape(x)[channel_axis] 

        unit_filters = int(in_channels/4)

        filters = {
            "branch1": unit_filters,
            "branch2_0": unit_filters,
            "branch2_1": unit_filters,
            "branch2_2": unit_filters}

    x0 = MaxPooling2D(pool_size=3, strides=2, padding="VALID")(x)

    x1 = conv_unit(x, filters=filters["branch1"], kernel_size=3, strides=2,
                   padding="VALID", order=order, **kargs)

    x2 = conv_unit(x, filters=filters["branch2_0"], kernel_size=1, strides=1,
                   padding="SAME", order=order, **kargs)
    x2 = conv_unit(x2, filters=filters["branch2_1"], kernel_size=3, strides=1,
                   padding="SAME", order=order, **kargs)
    x2 = conv_unit(x2, filters=filters["branch2_2"], kernel_size=3, strides=2,
                   padding="VALID", order=order, **kargs)

    filter_concat = Concatenate(axis=channel_axis)([x0, x1, x2])

    return filter_concat


def reduction_b(x,
                filters=None,
                order=["conv", "bn", "activation"],
                **kargs):
    """
    for 17 x 17 to 8 x 8 grid-reduction module. Reduction-B module used by the
    wider Inception-ResNet-v1 network.

    In: [1024, 17, 17]
    Out: [1536, 8, 8]
    """
    channel_axis = layer_utils.get_channel_axis()
    if filters is None:
        in_channels = K.int_shape(x)[channel_axis] 

        unit_filters = int(in_channels/4)

        filters = {
            "branch1_0": unit_filters,
            "branch1_1": int(1.5*unit_filters),
            "branch2_0": unit_filters,
            "branch2_1": unit_filters,
            "branch3_0": unit_filters,
            "branch3_1": unit_filters,
            "branch3_2": unit_filters}


    x0 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="VALID")(x)

    x1 = conv_unit(x, filters=filters["branch1_0"], kernel_size=1, strides=1,
                   padding="SAME", order=order, **kargs)
    x1 = conv_unit(x1, filters=filters["branch1_1"], kernel_size=3, strides=2,
                   padding="VALID", order=order, **kargs)

    x2 = conv_unit(x, filters=filters["branch2_0"], kernel_size=1, strides=1,
                   padding="SAME", order=order, **kargs)
    x2 = conv_unit(x2, filters=filters["branch2_1"], kernel_size=3, strides=2,
                   padding="VALID", order=order, **kargs)

    x3 = conv_unit(x, filters=filters["branch3_0"], kernel_size=1, strides=1,
                   padding="SAME", order=order, **kargs)
    x3 = conv_unit(x3, filters=filters["branch3_1"], kernel_size=3, strides=1,
                   padding="SAME", order=order, **kargs)
    x3 = conv_unit(x3, filters=filters["branch3_2"], kernel_size=3, strides=2,
                   padding="VALID", order=order, **kargs)

    filter_concat = Concatenate(axis=channel_axis)([x0, x1, x2, x3])

    return filter_concat

if __name__ == "__main__":
    from keras.layers import Input

    x = Input((256, 33, 33))

    red_a = reduction_a(x)
    red_b = reduction_b(red_a)
