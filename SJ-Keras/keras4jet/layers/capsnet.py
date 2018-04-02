from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K
import tensorflow as tf
from keras import initializers, models
from keras.engine.topology import Layer

import numpy as np

def squash(capsules):
    """ 
    inputs: [BatchSize, NumCaps, DimCaps]
    """
    # squared L2 norm
    squared_norm = K.sum(K.square(capsules), axis=-1, keepdims=True)
    scale = squared_norm / (1 + squared_norm) / K.sqrt(squared_norm + K.epsilon())
    return scale * capsules

class Length(Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]



class Masking(Layer):
    def call(self, inputs, training=None, **kargs):

        def masked_out_with_y_true():
            capsule, _, y_true = inputs
            return K.batch_flatten(capsule * K.expand_dims(y_true, -1))
    
        def masked_out_with_y_pred():
            capsule, y_pred, _ = inputs
            num_classes = K.int_shape(y_pred)[-1]
            mask = K.one_hot(
                indices=K.argmax(y_pred, 1), 
                num_classes=num_classes)
            return K.batch_flatten(capsule * K.expand_dims(mask, -1))

        return K.in_train_phase(
            x=masked_out_with_y_true,
            alt=masked_out_with_y_pred,
            training=training)

    def compute_output_shape(self, input_shape):
        return tuple([None, input_shape[0][1] * input_shape[0][2]])


class CapsuleLayer(Layer):
    """
    Ref. https://github.com/XifengGuo/CapsNet-Keras
    """
    def __init__(self, num_capsule, dim_capsule, num_routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.num_routings = num_routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"

        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(
            shape=[self.num_capsule,
                   self.input_num_capsule,
                   self.dim_capsule,
                   self.input_dim_capsule],
                   initializer=self.kernel_initializer,
            name='W')

        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 1)

        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.num_routings > 0, 'The routings should be > 0.'
        for i in range(self.num_routings):
            c = tf.nn.softmax(b, dim=1)

            s = K.batch_dot(c, inputs_hat, [2, 2])

            v = squash(s)

            if i < self.num_routings - 1:
                b += K.batch_dot(v, inputs_hat, [2, 3])

        return v

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            "num_capsule": self.num_capsule,
            "dim_capsule": self.dim_capsule,
            "num_routings": self.num_routings
        }

        config.update(super(CapsuleLayer, self).get_config())

        return config



