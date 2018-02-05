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
from keras.layers import Add
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.layers import Lambda
from keras.utils import conv_utils
import keras.backend as K

from keras4jet.layers import layer_utils
from keras4jet.layers.layer_utils import conv_unit



def _inception_resnet(x, filter_concat, filters, scaling_factor, activation="relu"):
    """
    * filter-expansion layer

    * We found that scaling down the residuals before adding them to the
      previous layer activation seemed to stabilize the training. In general we
      picked some scaling factors between 0.1 and 0.3 to scale the residuals
      before their being added to the accumulated layer activations
    """
    assert scaling_factor > 0 and scaling_factor <= 1


    filter_expansion = Conv2D(filters=filters, kernel_size=1)(filter_concat) 

    activation_scaling = Lambda(lambda tensor, scaling_factor: scaling_factor * tensor,
        output_shape=K.int_shape(filter_expansion)[1:],
        arguments={'scaling_factor': scaling_factor})(filter_expansion)

    # y_l = h(x_l) + F(x_l, W_l)
    y = Add()([x, activation_scaling])

    # x_{l+1} = f(y_l)
    x = Activation(activation)(y)

    return x

def inception_resnet_a(x, filters=None, scaling_factor=0.1):
    """
    for 35 x 35 grid (Inception-ResNet-A) module of Inception-ResNet-v1 network

    Input shape: [256, 35, 35]
    Output shape: [256, 35, 35]
    """
    channel_axis = layer_utils.get_channel_axis()
    if filters is None:
        in_channels = K.int_shape(x)[channel_axis] 

        unit_filters = int(in_channels/8)

        filters = {
            "branch0": unit_filters,
            "branch1_0": unit_filters,
            "branch1_1": unit_filters,
            "branch2_0": unit_filters,
            "branch2_1": unit_filters,
            "branch2_2": unit_filters,
            "filter_expansion": in_channels
        }


    x0 = conv_unit(x, kernel_size=1, filters=filters["branch0"])

    x1 = conv_unit(x, kernel_size=1, filters=filters["branch1_0"])
    x1 = conv_unit(x1, kernel_size=3, filters=filters["branch1_1"])


    x2 = conv_unit(x, kernel_size=1, filters=filters["branch2_0"])
    x2 = conv_unit(x2, kernel_size=3, filters=filters["branch2_1"])
    x2 = conv_unit(x2, kernel_size=3, filters=filters["branch2_2"])

    filter_concat = Concatenate(axis=channel_axis)([x0, x1, x2])

    out = _inception_resnet(
        x, filter_concat, filters["filter_expansion"], scaling_factor)

    return out




def inception_resnet_b(x, filters=None, scaling_factor=0.1):
    """
    for 17 x 17 grid (Inception-ResNet-B) module of Inception-ResNet-v1 network

    Input shape: [896, 17, 17]
    Output shape: [896, 17, 17]
    """
    channel_axis = layer_utils.get_channel_axis()

    if filters is None:
        in_channels = K.int_shape(x)[channel_axis]
        filters = {
            "branch0": int(in_channels / 7),
            "branch1_0": int(in_channels / 7),
            "branch1_1": int(in_channels / 7),
            "branch1_2": int(in_channels / 7),
            "filter_expansion": in_channels
        }

    x0 = conv_unit(x, kernel_size=1, filters=filters["branch0"])

    x1 = conv_unit(x, kernel_size=(1, 1), filters=filters["branch1_0"])
    x1 = conv_unit(x, kernel_size=(7, 1), filters=filters["branch1_1"])
    x1 = conv_unit(x, kernel_size=(1, 7), filters=filters["branch1_2"])

    filter_concat = Concatenate(axis=channel_axis)([x0, x1])

    out = _inception_resnet(x, filter_concat, filters["filter_expansion"], scaling_factor)
    return out 


def inception_resnet_c(x, filters=None, scaling_factor=0.1):
    """
    8x8 grid (Inception-ResNet-C) moduleof Inception-ResNet-v1 network.

    In: [1792, 8, 8]
    Out: [1792, 8, 8]
    """
    channel_axis = layer_utils.get_channel_axis()

    if filters is None:
        in_channels = K.int_shape(x)[channel_axis]
        unit_filters = int( ( 3 / 28 ) * in_channels )
        filters = {
            "branch0": unit_filters,
            "branch1_0": unit_filters,
            "branch1_1": unit_filters,
            "branch1_2": unit_filters,
            "filter_expansion": in_channels
        }

    x0 = conv_unit(x, kernel_size=1, filters=filters["branch0"])

    x1 = conv_unit(x, kernel_size=(1, 1), filters=filters["branch1_0"])
    x1 = conv_unit(x, kernel_size=(1, 3), filters=filters["branch1_1"])
    x1 = conv_unit(x, kernel_size=(3, 1), filters=filters["branch1_2"])

    filter_concat = Concatenate(axis=channel_axis)([x0, x1])

    out = _inception_resnet(x, filter_concat, filters["filter_expansion"], scaling_factor)
    return out 


if __name__ == "__main__":
    from keras.layers import Input
    from keras.models import Model
    from keras.utils import plot_model

    x = Input((256, 33, 33))

    res_a = inception_resnet_a(x)
    res_b = inception_resnet_b(res_a)
    res_c = inception_resnet_c(res_b)


    model = Model(inputs=x, outputs=res_c)

    plot_model(model, to_file="/tmp/Inception-ResNet.png")
