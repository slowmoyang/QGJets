from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import keras.backend as K

from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Reshape
from keras.layers import Permute 
from keras.layers import Dense
from keras.layers import Lambda

from keras.models import Model

from keras4jet.layers.capsnet import CapsuleLayer
from keras4jet.layers.capsnet import Length
from keras4jet.layers.capsnet import Masking
from keras4jet.layers.capsnet import squash

import tensorflow as tf
import numpy as np

def build_a_capsnet(input_shape, num_classes, num_routings):
    # Input tensor
    x = Input(shape=input_shape)

    # Conv1
    conv1 = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # PrimaryCaps
    dim_capsule = 8 
    num_channels = 32

    primary_caps = Conv2D(
        filters=dim_capsule*num_channels,
        kernel_size=9,
        strides=2,
        padding="valid",
        name='PrimaryCaps_ConvNode')(conv1)

    _, c, h, w = K.int_shape(primary_caps)

    primary_caps = Reshape(
        target_shape=(num_channels, dim_capsule, h, w), 
        name="PrimaryCaps_Reshape")(primary_caps) 

    # [batch_size, num_channels, dim_capsule, h, w] --> [batch_size, num_channels, h, w, dim_capsule]
    # [0, 1, 2, 3, 4] --> [0, 1, 3, 4, 2]
    primary_caps = Permute(dims=(1, 3, 4, 2), name="PrimaryCaps_Permute")(primary_caps)

    primary_caps = Reshape(target_shape=[-1, dim_capsule], name='PrimaryCaps_FlattenNode')(primary_caps)

    primary_caps = Lambda(squash, name='PrimaryCap_ActivationNode')(primary_caps)

    # DigitCaps
    digit_caps = CapsuleLayer(
        num_capsule=num_classes,
        dim_capsule=16,
        num_routings=num_routings,
        name='DigitCaps')(primary_caps)

    # Predictions
    y_pred = Length(name='Prediction')(digit_caps)

    #
    capsnet = Model(
        inputs=x,
        outputs=[y_pred, digit_caps],
        name="CapsNet")

    return capsnet



def build_a_decoder(input_shape, num_classes, out_caps_shape):
    out_caps = Input(shape=out_caps_shape, name="OutputCapsules")
    y_pred = Input(shape=(num_classes,), name="Prediction")
    y_true = Input(shape=(num_classes,), name="Label")


    # output_tensor = node(inptu_tensor)
    masked = Masking(name="MaskingNode")([out_caps, y_pred, y_true])

    decoding0 = Dense(512, activation="relu", name="DecodingNode0")(masked)
    decoding1 = Dense(1024, activation="relu", name="DecodingNode1")(decoding0)
    decoding2 = Dense(np.prod(input_shape), activation="relu", name="DecodingNode2")(decoding1)
    reconstructed = Reshape(target_shape=input_shape, name="Reconstruction")(decoding2)


    decoder = Model(
        inputs=[out_caps, y_pred, y_true],
        outputs=reconstructed,
        name="Decoder")


    return decoder


def build_a_model(input_shape, num_classes, num_routings):
    # Input tensor (placeholder)
    x = Input(input_shape, name="InputFeatures")
    y_true = Input((num_classes,), name="Label")

    # BUild a CapsNet
    caps_net = build_a_capsnet(input_shape, num_classes, num_routings)

    y_pred, digit_caps = caps_net(x)

    # Build a Decoder
    decoder = build_a_decoder(
        input_shape=input_shape,
        out_caps_shape=K.int_shape(digit_caps)[1:],
        num_classes=num_classes)

    reco = decoder([digit_caps, y_pred, y_true])


    model = Model(
        inputs=[x, y_true],
        outputs=[y_pred, reco],
        name="Model")

    return model




if __name__ == "__main__":
    from keras.utils import plot_model

    model = build_a_model((1,28,28), 10, 3)
    model.summary()

    plot_model(model, to_file="model.png")
