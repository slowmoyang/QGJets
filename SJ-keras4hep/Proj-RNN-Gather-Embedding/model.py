from collections import defaultdict

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers

from keras4hep.layers import Gather

_OP_COUNT = defaultdict(lambda: -1)

def _get_name(model_name, op_name):
    _OP_COUNT[op_name] += 1
    return "{}_{}_{}".format(model_name, op_name, _OP_COUNT[op_name])


def conv1d_block(tensor,
                 filters,
                 model_name,
                 activation="elu",
                 first=False):

    def get_name(op_name):
        return _get_name(model_name, op_name)

    residual = tensor

    if not first:
        tensor = Conv1D(filters,
                        kernel_size=1,
                        strides=1,
                        name=get_name("conv1d"))(tensor)

    tensor = BatchNormalization(
        axis=-1,
        name=get_name("batch_norm"))(tensor)

    tensor = Activation(activation, name=get_name(activation))(tensor)

    tensor = Conv1D(filters,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    name=get_name("conv1d"))(tensor)

    tensor = Concatenate(axis=2,
                         name=get_name("concat"))([tensor, residual])

    return tensor


def dense_block(tensor,
                units,
                activation="elu",
                model_name="rnn"):

    def get_name(op_name):
        return _get_name(model_name, op_name)


    tensor = Dense(units,
                   use_bias=False,
                   kernel_regularizer=regularizers.l2(0.01),
                   name=get_name("dense"))(tensor)

    tensor = BatchNormalization(
        axis=-1,
        name=get_name("batch_norm"))(tensor)


    tensor = Activation(
        activation,
        name=get_name(activation))(tensor)

    return tensor


def build_model(x_kin_shape,
                x_pid_shape,
                rnn="gru",
                activation="elu",
                name="rnn"):
    def get_name(op_name):
        return _get_name(name, op_name)

    x_kin = Input(x_kin_shape, name=get_name("input"))
    x_pid = Input(x_pid_shape, name=get_name("input"))
    x_len = Input(batch_shape=(None, 1), dtype=tf.int32, name=get_name("input"))

    h_kin = conv1d_block(tensor=x_kin, filters=64, activation=activation, first=True, model_name=name)

    h_pid = Embedding(input_dim=8, output_dim=8, name=get_name("embedding"))(x_pid)

    h = Concatenate(axis=-1, name=get_name("concat"))([h_kin, h_pid])

    rnn = rnn.lower()
    if rnn == "gru":
        h = GRU(units=128,
                return_sequences=True,
                name=get_name("gru"))(h)
#                 kernel_regularizer=regularizers.l2(0.01),
    elif rnn == "lstm":
        h = LSTM(units=128,
                 return_sequences=True,
                 name=get_name("lstm"))(h)
    else:
        raise ValueError

    h = Gather(name=get_name("gather"))([h, x_len])

    h = dense_block(h, 128, activation=activation, model_name=name)

    logits = Dense(units=2, name=get_name("dense"))(h)
    y = Softmax(name=get_name("softmax"))(logits)

    model = Model(inputs=[x_kin, x_pid, x_len],
                  outputs=y,
                  name=name)

    return model


def _test():
    from dataset import get_data_iter
    from keras4hep.projects.qgjets.utils import get_dataset_paths
    path = get_dataset_paths(min_pt=100)["training"]


    data_iter = get_data_iter(path, seq_maxlen={"x_kin": 32, "x_pid": 32})

    batch = data_iter.next()
    x_kin_shape = data_iter.get_shape("x_kin", batch_shape=False)
    x_pid_shape = data_iter.get_shape("x_pid", batch_shape=False)

    model = build_model(x_kin_shape, x_pid_shape, name="RNN")
    model.summary()

    y_score = model.predict_on_batch([batch.x_kin, batch.x_pid, batch.x_len])
    print("y_score: {}".format(y_score.shape))


if __name__ == "__main__":
    _test()
