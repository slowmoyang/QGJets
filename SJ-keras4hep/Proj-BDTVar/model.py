import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU

from keras4hep.data import DataIterator
from dataset import BDTVarSet



def dense_block(x, units, activation="relu"):
    x = Dense(units, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def build_a_model(x_shape, activation="elu"):
    x = Input(x_shape)

    h = dense_block(x, 64, activation=activation)
    h = dense_block(h, 128, activation=activation)
    h = dense_block(h, 32, activation=activation)
    logits = Dense(units=2)(h)
    y = Activation("softmax")(logits)

    model = Model(
        inputs=x,
        outputs=y)

    return model

def _test():
    path = "/store/slowmoyang/QGJets/dijet_100_110/dijet_100_110_test.root"
    dset = BDTVarSet(path)
    data_iter = DataIterator(dset, batch_size=128)

    batch = data_iter.next()
    x_shape = data_iter.get_shape("x", batch_shape=False)

    model = build_a_model(x_shape)

    logits = model.predict_on_batch(batch.x)
    print("logits: {}".format(logits.shape))


if __name__ == "__main__":
    _test()
