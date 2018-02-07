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
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D

from keras4jet.layers import layer_utils
from keras4jet.layers.layer_utils import conv_unit

from keras4jet.layers.inception import inception_resnet_a
from keras4jet.layers.inception import inception_resnet_b
from keras4jet.layers.inception import inception_resnet_c
from keras4jet.layers.inception import reduction_a
from keras4jet.layers.inception import reduction_b

def stem(x):
    channel_axis = layer_utils.get_channel_axis()

    x = conv_unit(x, filters=32, kernel_size=7)

    x_pool = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="VALID")(x)
    x_conv = conv_unit(x, filters=96, kernel_size=(3,3), strides=(2,2), padding="VALID")

    x = Concatenate(axis=channel_axis)([x_pool, x_conv])

    x = conv_unit(x, filters=256, kernel_size=(1,1))
    return x

def build_a_model(input_shape,
                  num_classes=2,
                  num_a=1,
                  num_b=1,
                  num_c=1,
                  dropout_rate=0.2,
                  activation="relu",
                  padding="SAME"):
    # Input: [C, 33, 33]
    inputs = Input(input_shape)

    # Stem
    # [C, 33, 33] --> [?, 33, 33]
    out = stem(inputs)

    # Inception-ResNet-A
    # [?, 33, 33] --> [, 33, 33
    for _ in xrange(num_a):
        out = inception_resnet_a(out) 

    out = reduction_a(out)

    # Inception-ResNet-B
    for _ in xrange(num_b):
        out = inception_resnet_b(out) 

    # Reduction-B
    out = reduction_b(out)

    # Inception-ResNet-C
    for _ in range(num_c):
        out = inception_resnet_c(out)

    #
    out = GlobalAveragePooling2D()(out)

    out = Dropout(rate=dropout_rate)(out)

    out = Dense(units=num_classes)(out)

    out = Activation("softmax")(out)

    model = Model(inputs=inputs, outputs=out)
    return model

if __name__ == "__main__":
    from keras.utils import plot_model

    model = build_a_model((10, 33, 33))

    plot_model(
        model,
        show_shapes=True,
        to_file="/tmp/Inception-ResNet-v2.png")
