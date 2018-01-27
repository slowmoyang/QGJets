from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import backend as K

from keras.models import Model

from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D

import numpy as np
import tensorflow as tf

#if __name__ == "__main__":
#    import sys
#    sys.path.append("../../layers")
#    import layer_utils
#    from layer_utils import conv_unit
#    from layer_utils import factorized_conv
#    from densenet import dense_block
#    from densenet import transition_layer
#else:
from keras4jet.layers import layer_utils
from keras4jet.layers.layer_utils import conv_unit
from keras4jet.layers.layer_utils import factorized_conv
from keras4jet.layers.densenet import dense_block
from keras4jet.layers.densenet import transition_layer

def build_a_model(input_shape,
                  num_classes=2,
                  growth_rate=12, # k
                  theta=0.5, # compression_factor
                  num_layers_list=[6, 12, 32, 32],
                  use_bottleneck=True,
                  init_filters=None,
                  activation="relu"):
    """
    Args:
      input_shape: 'tuple', (channels, height, width).
      num_classes: 'int'. Default is 2.
      grwoth_rate: 'int'
    """

    if init_filters is None:
        if use_bottleneck and theta == 1:
            init_filters = 2 * grwoth_rate
        else:
            init_filters = 16

    # (B, C, 33, 33)
    inputs = Input(input_shape)

    # (B, C, 17, 17) 
    out = conv_unit(
        inputs,
        filters=init_filters,
        kernel_size=7,
        strides=2,
        padding="SAME",
        activation=activation,
        order=["bn", "activation", "conv"])

    num_blocks = len(num_layers_list)
    for i, num_layers in enumerate(num_layers_list, 1):
        out = dense_block(out, num_layers, growth_rate, use_bottleneck, activation)
        if i  < num_blocks:
            out = transition_layer(out, theta)


    #
    #At the end of the last dense block, a global average pooling is performed
    # and then a softmax classifier is attached.
    out = GlobalAveragePooling2D()(out)
    out = Dense(units=num_classes)(out)
    out = Activation("softmax")(out)

    model = Model(inputs=inputs, outputs=out)
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    model = build_a_model(
        input_shape=(1,33,33),
        num_classes=2)

    plot_model(model, to_file="/home/slowmoyang/Lab/QGJets/Keras/keras4jet/models/image/model_plot/densenet.png", show_shapes=True)  
