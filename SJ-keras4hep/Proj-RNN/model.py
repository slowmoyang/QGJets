import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU


def conv1d_block(x, filters, kernel_size=1, strides=1, activation="elu"):
    x = Conv1D(filters, kernel_size, strides=strides)(x)
    x = Activation(activation)(x)
    return x


def dense_block(x, units, activation="elu"):
    x = Dense(units)(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis=-1)(x)
    return x


def build_a_model(x_shape):
    x = Input(x_shape)

    h = conv1d_block(x, 128)

    h = GRU(units=256)(h)

    h = dense_block(h, 32)
    logits = Dense(units=2)(h)
    y = Activation("softmax")(logits)

    model = Model(
        inputs=x,
        outputs=y)

    return model

def _test():
    from keras4hep.data import DataIterator
    from dataset import JetSeqSet

    path = "/store/slowmoyang/QGJets/dijet_100_110/dijet_100_110_test.root"
    prep_path = "./logs/dijet_100_110_training.npz"
    dset = JetSeqSet(path, extra=["pt", "eta"],
                     seq_maxlen={"x": 50})
    data_iter = DataIterator(dset, batch_size=128)

    batch = data_iter.next()
    x_shape = data_iter.get_shape("x", batch_shape=False)

    model = build_a_model(x_shape)

    y_score = model.predict_on_batch([batch.x])
    print("y_score: {}".format(y_score.shape))


if __name__ == "__main__":
    _test()
