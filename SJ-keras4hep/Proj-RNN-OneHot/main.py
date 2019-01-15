from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras4hep as kh
from keras4hep.data import DataIterator
from keras4hep.projects.qgjets import QGJetsExperiment

from dataset import PIDOneHotSet
from model import build_model as model_fn

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
        parser.add_argument("--rnn", default="gru", type=str)
        return parser

    def build_model(self):
        x_shape = self.train_iter.get_shape("x")

        self.model = model_fn(
            x_shape=x_shape,
            rnn=self.config.rnn)


def main():
    expt = RNNOneHotExpt()
    expt.run()


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
