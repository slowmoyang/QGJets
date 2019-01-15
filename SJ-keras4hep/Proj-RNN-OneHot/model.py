from collections import defaultdict

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers

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


def build_model(x_shape,
                rnn="gru",
                activation="elu",
                name="rnn"):
    def get_name(op_name):
        return _get_name(name, op_name)

    x = Input(x_shape, name=get_name("input"))
    h = conv1d_block(
        tensor=x,
        filters=64,
        activation=activation,
        first=True,
        model_name=name)
    h = GRU(units=128, name=get_name("gru"))(h)


    h = dense_block(h, 128, activation=activation, model_name=name)

    logits = Dense(units=2, name=get_name("dense"))(h)
    y = Softmax(name=get_name("softmax"))(logits)

    inputs = [x]
    outputs = [y]

    model = Model(inputs=inputs,
                  outputs=outputs,
                  name=name)

    return model


def get_custom_objects():
    from keras4hep.metrics import roc_auc
    custom_objects = {
        "roc_auc": roc_auc
    }
    return custom_objects



def _test():
    from dataset import get_data_iter
    from keras4hep.projects.qgjets.utils import get_dataset_paths
    path = get_dataset_paths(min_pt=100)["training"]

    seq_maxlen = {
        "x": (40, "float32"),
    }
    data_iter = get_data_iter(path, seq_maxlen=seq_maxlen)

    batch = data_iter.next()
    x_shape = data_iter.get_shape("x", batch_shape=False)

    model = build_model(x_shape, name="RNN")
    model.summary()

    y_score = model.predict_on_batch([batch.x])
    print("y_score: {}".format(y_score.shape))


if __name__ == "__main__":
    _test()
