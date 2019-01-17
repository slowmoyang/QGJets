from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import keras4hep as kh
from keras4hep.data import DataIterator
from keras4hep.projects.qgjets import QGJetsExperiment

from dataset import PIDOneHotSet
from model import build_classifier


class RNNOneHotExpt(QGJetsExperiment):
    def __init__(self):
        super(RNNOneHotExpt, self).__init__()

        self.config["seq_maxlen"] = {
            "x": (self.config.seq_maxlen, "float32"),
        }

    def get_dataset(self, path):
        dataset = PIDOneHotSet(path=path, seq_maxlen=self.config.seq_maxlen)
        return dataset

    def make_argument_parser(self):
        parser = super(RNNOneHotExpt, self).make_argument_parser()
        parser.add_argument("--maxlen", dest="seq_maxlen", default=30, type=int)
        return parser

    def build_model(self):
        x_shape = self.train_iter.get_shape("x")

        self.model = build_classifier(x_shape=x_shape)


def main():
    expt = RNNOneHotExpt()
    expt.run()


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # tf.logging.set_verbosity(tf.logging.WARN)

    import matplotlib as mpl
    mpl.use('Agg')

    main()
