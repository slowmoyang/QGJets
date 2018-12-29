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
                 filters=None,
                 activation="elu",
                 first=False,
                 model_name="rnn"):

    def get_name(op_name):
        return _get_name(model_name, op_name)

    if filters is None:
        in_features = K.int_shape(x)[-1]
        filters = int(in_features / 2)

    residual = tensor

    if not first:
        tensor = Conv1D(filters,
                        kernel_size=1,
                        strides=1,
                        kernel_regularizer=regularizers.l2(0.01),
                        name=get_name("conv1d"))(tensor)

    tensor = Activation(activation, name=get_name(activation))(tensor)

    tensor = BatchNormalization(
        axis=-1,
        momentum=0.9,
        name=get_name("batch_normalization"))(tensor)

    tensor = Conv1D(filters,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    kernel_regularizer=regularizers.l2(0.01),
                    name=get_name("conv1d"))(tensor)

    tensor = Concatenate(axis=2,
                         name=get_name("concatenate"))([tensor, residual])

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
        momentum=0.9,
        name=get_name("batch_normalization"))(tensor)


    tensor = Activation(
        activation,
        name=get_name(activation))(tensor)

    return tensor


def build_model(x_kin_shape,
                x_pid_shape,
                activation="elu",
                name="rnn"):
    def get_name(op_name):
        return _get_name(name, op_name)

    x_kin = Input(x_kin_shape, name=get_name("input"))
    x_pid = Input(x_pid_shape, name=get_name("input"))
    x_len = Input(batch_shape=(None, 1),
                  dtype=tf.int32,
                  name=get_name("input"))

    h_kin = conv1d_block(tensor=x_kin, filters=256, activation=activation, first=True, model_name=name)

    h_pid = Embedding(input_dim=8, output_dim=64, name=get_name("embedding"))(x_pid)
    # h_pid = conv1d_block(h_pid, filters=128, activation=activation, first=True, model_name=name)

    h = Concatenate(axis=-1, name=get_name("concatenate"))([h_kin, h_pid])
    # h = Concatenate(axis=-1, name=get_name("concatenate"))([x_kin, h_pid])

    h = GRU(units=512,
             return_sequences=True,
             kernel_regularizer=regularizers.l2(0.01),
             name=get_name("gru"))(h)

    h = Gather(name=get_name("gather"))([h, x_len])

    h = dense_block(h, 256, activation=activation, model_name=name)
    h = dense_block(h, 64, activation=activation, model_name=name)
    h = dense_block(h, 32, activation=activation, model_name=name)

    logits = Dense(units=2, name=get_name("dense"))(h)
    y = Activation("softmax", name=get_name("softmax"))(logits)

    model = Model(inputs=[x_kin, x_pid, x_len],
                  outputs=y,
                  name=name)

    return model


def get_custom_objects():
    from keras4hep.metrics import roc_auc
    custom_objects = {
        "Gather": Gather,
        "tf": tf,
        "roc_auc": roc_auc
    }
    return custom_objects



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
