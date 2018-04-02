from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.layers import Dropout
from keras.layers import Wrapper

import numpy as np


class MCDropout(Dropout):
    """
    Why input is scaled in tf.nn.dropout in tensorflow?
    https://stackoverflow.com/questions/34597316/why-input-is-scaled-in-tf-nn-dropout-in-tensorflow
    """
    def call(self, inputs):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            outputs = K.dropout(
                inputs,
                self.rate,
                noise_shape,
                seed=self.seed)
        else:
            outputs = inputs

        return outputs




class ConcreteDropout(Wrapper):
    """
    Ref. https://github.com/yaringal/ConcreteDropout
    """
    def __init__(self,
                 layer,
                 weight_regularizer=1e-6,
                 dropout_regularizer=1e-5,
                 init_min=0.1,
                 init_max=0.1,
                 temperature=0.1,
                 is_mc_dropout=True,
                 **kwargs):
        assert not kwargs.has_key("kernel_regularizer")
        super(ConcreteDropout, self).__init__(layer, **kwargs)

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout

        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

        self.temperature = temperature

        self.supports_masking = True
        self.p_logit = None
        self.p = None

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()

        # initialise p
        self.p_logit = self.layer.add_weight(
            name="p_logit",
            shape=(1, ),
            initializer=initializers.RandomUniform(
                self.init_min,
                self.init_max),
            trainable=True)

        self.p = K.sigmoid(self.p_logit[0])

        # Initialise regulariser / prior KL term
        input_dim = np.prod(input_shape[1:]) # We drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1.0 - self.p)

        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim

        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)


    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        """Concrete dropout
        Args:
          x:
        Returns:
          A
        """
        eps = K.cast_to_floatx(K.epsilon())

        # Unifrom noise
        u = K.random_uniform(shape=K.shape(x))

        # the Concrete distribution relaxation 
        z_tilde = (
            K.log(self.p + eps)
            - K.log(1.0 - self.p + eps)
            + K.log(u + eps)
            - K.log(1.0 - u + eps)
        )
        z_tilde = K.sigmoid(z_tilde / self.temperature)

        # Random tensor
        random_tensor = 1.0 - drop_prob

        retain_prob = 1.0 - self.p

        x *= random_tensor
        x /= retain_prob

        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(
                x=relaxed_dropped_inputs,
                alt=self.layer.call(inputs),
                training=training)


def compute_init_regularizer(num_instances, l=1e-4):
    """
    Args:
      l: A float. prior lengthscale
    """
    wd = l**2.0 / num_instances
    dd = 2.0 / num_instances 
    return wd, dd
