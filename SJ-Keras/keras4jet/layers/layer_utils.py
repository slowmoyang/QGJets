from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.utils import conv_utils
import keras.backend as K

def get_channel_axis():
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    return channel_axis



def conv_unit(x,
              filters,
              kernel_size,
              activation="relu",
              order=["conv", "bn", "activation"],
              **kargs):
    if isinstance(order, str):
        order = order.lower().replace(" ", "_")
        if order in ["full_pre-activation"]:
            order = ["bn", "activation", "conv"]
        else:
            raise ValueError("")


    channel_axis = get_channel_axis()

    conv_keys = ["strides", "padding", "dilation_rate", "activation",
                 "use_bias", "kernel_initializer", "bias_initializer",
                 "kernel_regularizer", "bias_regularizer",
                 "activity_regularizer", "kernel_constraint",
                 "bias_constraint"]
        
    bn_keys = ["momentum", "epsilon", "center", "scale", "beta_initializer",
               "gamma_initializer", "moving_mean_initializer",
               "moving_variance_initializer", "beta_regularizer",
               "gamma_regularizer", "beta_constraint", "gamma_constraint"]

    conv_kargs = {}
    bn_kargs = {}
    activation_kargs ={}

    if kargs.has_key("name"):
        name = kargs.pop("name")
        conv_kargs["name"] = name + "_Conv2D"
        bn_kargs["name"] = name + "_BatchNorm"
        activation_kargs["name"] = name + "_Activation"

    for key in kargs.keys():
        if key in conv_keys:
            conv_kargs[key] = kargs[key]
        elif key in bn_keys:
            bn_kargs[key] = kargs[key]
        else:
            raise ValueError(":p") 

    if not conv_kargs.has_key("padding"):
        conv_kargs["padding"] = "same"


    node_dict = {
        "conv": Conv2D(filters, kernel_size, **conv_kargs),
        "bn": BatchNormalization(axis=channel_axis, **bn_kargs),
        "activation": Activation(activation=activation, **activation_kargs)
    }
    
    for each in order:
        x = node_dict[each](x)

    return x

def factorized_conv(x, filters, kernel_size, strides=(1, 1), asym=True, **kargs):
    kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    strides = conv_utils.normalize_tuple(strides, 2, 'strides')

    assert kernel_size[0] == kernel_size[1]
    k = kernel_size[0]

    assert bool(k%2)

    if asym:
        x = conv_unit(x, filters, kernel_size=(1, k), strides=strides[1], **kargs)
        x = conv_unit(x, filters, kernel_size=(k, 1), strides=strides[0], **kargs)
    else:
        num_conv = int((k - 1) / 2)
        for _ in range(num_conv):
            x = conv_unit(x, filters, kernel_size=(3,3), strides=strides, **kargs)
    return x




