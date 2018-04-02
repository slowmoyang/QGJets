from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


from keras.engine.training import Model


class BayesianModel(Model):
    def __init__(self, inputs, outputs, tau, name=None):
        super(BayesianModel, self).__init__(inputs, outputs, name)
        self.tau = tau

    def stochastic_forward_pass(self, x, T=1000):
        probs = np.array([self.predict_on_batch(x) for _ in xrange(T)])

        pred_mean = np.mean(probs, axis=0)
        pred_var = np.var(probs, axis=0) + self.tau**-1

        return (pred_mean, pred_var)
