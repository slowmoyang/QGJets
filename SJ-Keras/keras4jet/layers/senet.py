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


from keras4jet.layers import layer_utils
from keras4jet.layers.layer_utils import conv_unit

from keras4jet.layers.resnet import residual_fn
from keras4jet.layers.resnet import residual_unit


# SE block
def se_block(u, reduction_ratio=4, activation="relu"):
    """
    The Squeeze-and-Excitation block is a computational unit which can be
    constructed for any given transformation
        F_tr : X --> U, X in R^{W'xH'xC'}, U in R^{WxHxC}.


    u_c  = F_tr(X) = v_c * X
    z_c  = F_sq(u_c)
    s    = F_ex(z, W)
    x~_c = F_scale(u_c, s_c)

    F_tr : a standard convolutional operator.
    F_sq : 

    """
    channel_axis = layer_utils.get_channel_axis()
    in_channels = K.int_shape(u)[channel_axis]
    bottleneck_units = int(in_channels / reduction_ratio)

    if K.image_data_format() == "channels_first":
        excitation_reshape = (-1, 1, 1)
    else:
        excitation_reshape = (1, 1, -1)

    # Squeeze: Global Information Embedding
    z = GlobalAveragePooling2D()(u)

    # Excitation: F_ex
    s = Dense(bottleneck_units)(z)
    s = Activation(activation)(s)
    s = Dense(in_channels)(s)
    s = Activation("sigmoid")(s)
    s = Reshape(excitation_reshape)(s)

    # Scale: F_scale
    x_tilde = Multiply()([s, u])
    return x_tilde


def se_resnet_module(x, filters, reduction_ratio=4, **kargs):
    def _residual_function(x, filters):
        u = residual_fn(x, filters, **kargs) 
        x_tilde = se_block(u, reduction_ratio)
        return x_tilde

    y = residual_unit(
        x=x,
        filters=filters,
        residual_fn=_residual_function,


    u = residual_unit(x, filters=filters, **kargs) 
    x_tilde = se_block(u, reduction_ratio, activation)

    y = Add()([x, x_tilde])
    return y


def se_inception_resnet(x, reduction_ratio=4, **kargs):
    pass
