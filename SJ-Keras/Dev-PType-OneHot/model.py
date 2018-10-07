import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU
from keras.layers import Input
from keras.layers import LeakyReLU

import sys
sys.path.append("..")
from keras4jet.data import DataIterator
from dataset import PTypeDataset

def conv1d_block(x, filters, kernel_size=1, strides=1):
    x = BatchNormalization()(x)
    x = Conv1D(filters, kernel_size, strides=strides)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def dense_block(x, units):
    x = BatchNormalization()(x)
    x = Dense(units)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def build_a_model(x_shape):
    x = Input(x_shape)

    h = conv1d_block(x, 32)
    h = GRU(units=128)(h)

    # h = Dropout(0.5)(h)
    h = dense_block(h, 32)
    logits = Dense(units=1)(h)
    y = Activation("sigmoid")(logits)

    model = Model(
        inputs=x,
        outputs=y)

    return model

def _test():
    path = "/store/slowmoyang/QGJets/data/root_100_200/2-Refined/dijet_test_set.root"
    dset = PTypeDataset(path, seq_maxlen={"x": 50})
    data_iter = DataIterator(dset, batch_size=128)

    batch = data_iter.next()
    x_shape = data_iter.get_shape("x", batch_shape=False)

    model = build_a_model(x_shape)

    logits = model.predict_on_batch(batch.x)
    print("logits: {}".format(logits.shape))


if __name__ == "__main__":
    _test()
