from __future__ import absolute_import

import os

import functools

import numpy as np

import keras
import keras.backend as K

class ReduceLROnPlateau(object):
    def __init__(self, model, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        self.model = model
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1

        self.lr_epsilon = self.min_lr * 1e-4


        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

        self._metrics = []

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics=None, epoch=None):
        if metrics is None:
            current = np.mean(self._metrics)
            self.clear_metrics()
        else:
            current = metrics

        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        old_lr = float(K.get_value(self.model.optimizer.lr))

        if old_lr > self.min_lr + self.lr_epsilon:

            new_lr = old_lr * self.factor

            new_lr = max(new_lr, self.min_lr)

            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose:
                print('\nEpoch %05d: reducing learning rate to %s.' % (epoch + 1, new_lr))
            self.cooldown_counter = self.cooldown


    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon
        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')
        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = -1 * float('inf')

        self.is_better = functools.partial(self._cmp, mode, threshold_mode, threshold)


    def monitor(self, metrics):
        self._metrics.append(metrics)

    def clear_metrics(self):
        self.metrics = []




if __name__ == "__main__":
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(model, verbose=True)

    epoch = 0
    print("(Epoch {epoch:03d}) lr: {lr}".format(
        epoch=epoch, lr=model.optimizer.lr))
    epoch += 1

    for epoch in xrange(100):
        reduce_lr.step(0.3)
        print("(Epoch {epoch:03d}) lr: {lr}".format(
            epoch=epoch, lr=model.optimizer.lr))

