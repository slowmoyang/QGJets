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

from keras4jet.layers import layer_utils

from keras4jet.layers.senet.senet import se_block

from keras4jet.layers.inception.inception_resnet import _residual_a
from keras4jet.layers.inception.inception_resnet import _residual_b
from keras4jet.layers.inception.inception_resnet import _residual_c
from keras4jet.layers.inception.inception_resnet import _inception_resnet
from keras4jet.layers.inception.reduction import reduction_a
from keras4jet.layers.inception.reduction import reduction_b

def se_inception_resnet_a(x, # 
                          reduction_ratio=4, # se_block
                          scaling_factor=0.1, # _inception_resnet
                          filters=None,
                          order=["conv", "bn", "activation"]):
    residual = _residual_a(x, filters, order)
    scale = se_block(residual, reduction_ratio)
    out = _inception_resnet(x, scale, scaling_factor)
    return x


def se_inception_resnet_b(x,
                          reduction_ratio=4, # se_block
                          scaling_factor=0.1, # _inception_resnet
                          filters=None,
                          order=["conv", "bn", "activation"]):
    residual = _residual_b(x, filters, order)
    scale = se_block(residual, reduction_ratio)
    out = _inception_resnet(x, scale, scaling_factor)
    return x


def se_inception_resnet_c(x,
                          reduction_ratio=4, # se_block
                          scaling_factor=0.1, # _inception_resnet
                          filters=None,
                          order=["conv", "bn", "activation"]):
    residual = _residual_c(x, filters, order)
    scale = se_block(residual, reduction_ratio)
    out = _inception_resnet(x, scale, scaling_factor)
    return x


def se_reduction_a(x, filters=None, order=["conv", "bn", "activation"], reduction_ratio=4):
    x = reduction_a(x, filters, order)
    x = se_block(x, reduction_ratio)
    return x

def se_reduction_b(x, filters=None, order=["conv", "bn", "activation"], reduction_ratio=4):
    x = reduction_b(x, filters, order)
    x = se_block(x, reduction_ratio)
    return x
