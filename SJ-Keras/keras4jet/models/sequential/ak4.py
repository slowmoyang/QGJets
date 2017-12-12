from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import keras
from keras.layers import (
    Input,
    Conv1D,
    LSTM,
    GRU,
    Concatenate,
    Dense,
    Activation,
    BatchNormalization
)
from keras.models import (
    Model,
    Sequential
)

def preact_block(x, filters, act_fn="relu"):
    x = BatchNormalization()(x)
    x = Activation(act_fn)(x)
    x = Conv1D(filters, 1, strides=1)(x)
    return x

def dense_block(x, units, act_fn="relu"):
    x = BatchNormalization()(x)
    x = Dense(units)(x)
    x = Activation(act_fn)(x)
    return x

def build_a_model(input0_shape, input1_shape):
    """
    input0_shape: x_daus_shape
    input1_shape: x_glob_shape
    """
    # Input features
    input_daus = Input(input0_shape)
    input_glob = Input(input1_shape)

    # 1x1 Conv Block
    # Conv1D(filters, kernel_size, strides, padding, dilation_rate, activation
    x = preact_block(input_daus, 128)
    x = preact_block(x, 96)
    x = preact_block(x, 80)
    x = preact_block(x, 20)
    
    # Recurrent layer
    x = LSTM(units=150)(x)

    # concat
    x = Concatenate()([x, input_glob])

    # Dense Block
    x = dense_block(x, 200)
    x = dense_block(x, 100)
    x = dense_block(x, 100)
    x = dense_block(x, 100)
    x = dense_block(x, 100)
    x = dense_block(x, 100)
    x = dense_block(x, 100)

    # Output: sigmoid
    sigmoid = Dense(units=1, activation='sigmoid')(x)

    # Create model.
    model = Model(
        inputs=[input_daus, input_glob],
        outputs=sigmoid,
        name="AK4Net")
    return model

