import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU

from keras4hep.data import DataIterator


def conv_block(x, filters, kernel_size=5, strides=(1, 1), activation="relu", padding="valid"):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis=1)(x)
    return x


def build_a_model(x_shape, filters_list=[16, 64, 32], activation="relu", top="proj"):
    x = Input(x_shape)

    h = x
    for filters in filters_list:
        h = conv_block(h, filters, activation=activation, padding="valid")

    if top == "dense":
        h = Flatten()(h)
        logits = Dense(2)(h)
    elif top == "proj":
        h = Conv2D(filters=2, kernel_size=1, strides=1)(h)
        logits = GlobalAveragePooling2D()(h)
    else:
        raise NotImplementedError

    y_score = Activation("softmax")(logits)

    model = Model(inputs=x,
                  outputs=y_score)

    return model

def _test():
    from dataset import C10Set
    path = "/store/slowmoyang/QGJets/dijet_100_110/dijet_100_110_test.root"
    dset = C10Set(path)
    data_iter = DataIterator(dset, batch_size=128)

    batch = data_iter.next()
    x_shape = data_iter.get_shape("x", batch_shape=False)

    model = build_a_model(x_shape)

    logits = model.predict_on_batch(batch.x)
    print("logits: {}".format(logits.shape))


if __name__ == "__main__":
    _test()
