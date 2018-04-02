from __future__ import absolute_import
from __future__ import division

import keras.backend as K


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))


def reco_loss(x, reco):
    return K.mean(K.square(x - reco), axis=[1, 2, 3])


def softmax_loss(y_true, y_pred):
    return -1 * K.mean(K.log(K.sum(y_true * y_pred, axis=-1) + K.epsilon()))

