from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

from keras import backend as K

from keras.models import Model

from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization 
from keras.layers import GlobalAveragePooling2D

import numpy as np
import tensorflow as tf

def conv_block(filters, kernel_size, strides, padding, activation):
    def _block(x):
        out = BatchNormalization()(x)
        out = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(out)
        out = Activation(activation)(out)
        return out
    return _block

def dense_block(units, node_order=["dense", "bn", "activation"], **kargs):
    activation = "relu" if not kargs.has_key("activation") else kargs["activation"]
    drop_rate = 0.5 if not kargs.has_key("drop_rate") else kargs["drop_rate"]

    def _block(x):
        node_order = [node.lower() for node in node_order]
        for node in node_order:
            if node == "dense":
                x = Dense(units, use_bias=use_bias)(x)
            elif node in ["bn", "batch_norm", "batch_normalization", "batchnormalization"]:
                x = BatchNormalization()(x)
            elif node == "activation":
                x = Activation(activation)(x)
            elif node == "dropout":
                x = Dropout(drop_rate)(x)
        return x

    return _block



def build_a_model(image_shape, # Conv
                  num_features, # Dense
                  num_classes=2,
                  units_list=[32, 64, 128], # Dense
                  filters_list=[64, 64, 128, 128],
                  last_hidden_units=32,
                  kernel_size=5, # Conv
                  drop_rate=0.0, # Dense
                  activation="relu",
                  padding="SAME"):



    #######################
    # CNN Stem
    ######################
    x_image = Input(image_shape)
    h_conv = Conv2D(filters=filters_list[0], kernel_size=kernel_size, strides=2, padding=padding)(x_image)
    h_conv = Activation(activation)(h_conv)

    for filters in filters_list[1:]:
        h_conv = conv_block(filters, kernel_size, 1, padding, activation)(h_conv)
        if (i != 0) and (i % 2 == 0):
            h_conv = MaxPooling2D(2)(h_conv)

    h_conv = GlobalAveragePooling2D()(h_conv)

    ##################################
    # DNN Stem
    ###################################
    x_features = Input((num_features, ))

    h_dense = x_features
    for units in units_list:
        h_dense = dense_block(units, activation, drop_rate=drop_rate)(h_dense)

    ######################################
    # Merge 
    ##########################################
    h_concat = Concatenate()([h_conv, h_dense])
    h_concat = dense_block(units=last_hidden_units, activation=activation)(h_concat)

    logits = Dense(units=num_classes)(h_concat)
    y_score = Activation("softmax")(logits)
    ############################
    #
    ############################
    model = Model(inputs=[x_image, x_features], outputs=y_score)

    return model

