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


def build_a_model(x_jet_shape,
                  x_constituents_shape):
    """
    """
    x_jet = Input(x_jet_shape)
    x_constituents = Input(x_constituents_shape)

    # x_jet
    h_j = dense_block(x_jet, 16)
    h_j = dense_block(x_jet, 32)
    h_j = dense_block(x_jet, 64)

    # x_con
    h_c = conv1d_block(x_constituents, 16)
    h_c = conv1d_block(x_constituents, 32)
    h_c = conv1d_block(x_constituents, 64)
    h_c = GRU(units=128, dropout=0.5, recurrent_dropout=0.5)(h_c)

    h = Concatenate()([h_j, h_c])

    h = Dropout(0.5)(h)
    h = dense_block(h, 32)
    logits = Dense(units=2)(h)
    y_pred = Activation("softmax")(logits)

    model = Model(
        inputs=[x_jet, x_constituents],
        outputs=y_pred)

    return model
