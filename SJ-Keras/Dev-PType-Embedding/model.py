import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
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
    x = Activation("elu")(x)
    return x


def build_a_model(x_kinematics_shape, x_pid_shape):
    x_kinematics = Input(x_kinematics_shape)
    x_pid = Input(x_pid_shape)

    pid_embedding = Embedding(input_dim=x_pid_shape[-1], output_dim=16)(x_pid)

    x = Concatenate()([x_kinematics, pid_embedding])

    h = conv1d_block(x, 32)
    h = GRU(units=128, dropout=0.5)(h)

    # h = Dropout(0.5)(h)
    h = dense_block(h, 32)
    logits = Dense(units=1)(h)
    y = Activation("sigmoid")(logits)

    model = Model(
        inputs=[x_kinematics, x_pid],
        outputs=y)

    return model

def _test():
    path = "/store/slowmoyang/QGJets/data/root_100_200/2-Refined/dijet_test_set.root"
    dset = PTypeDataset(
        path,
        seq_maxlen={
            "x_kinematics": 30,
            "x_pid": 30
        },
        extra=["pt", "eta"])
    data_iter = DataIterator(dset, batch_size=128)

    batch = data_iter.next()
    x_kinematics_shape = data_iter.get_shape("x_kinematics", batch_shape=False)
    x_pid_shape = data_iter.get_shape("x_pid", batch_shape=False)

    model = build_a_model(x_kinematics_shape, x_pid_shape)

    logits = model.predict_on_batch([batch.x_kinematics, batch.x_pid])
    print("logits: {}".format(logits.shape))


if __name__ == "__main__":
    _test()
