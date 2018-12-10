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


def conv1d_block(x,
                 filters=None,
                 activation="elu",
                 first=False,
                 model_name="rnn"):
    if filters is None:
        in_features = K.int_shape(x)[-1]
        filters = int(in_features / 2)

    residual = x

    if not first:
        x = Conv1D(filters,
                   kernel_size=1,
                   strides=1,
                   kernel_regularizer=regularizers.l2(0.01),
                   name=_get_name(model_name, "conv1d"))(x)

    x = Activation(activation, name=_get_name(model_name, activation))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv1D(filters,
               kernel_size=1,
               strides=1,
               use_bias=False,
               kernel_regularizer=regularizers.l2(0.01),
               name=_get_name(model_name, "conv1d"))(x)

    x = Concatenate(axis=2, name=_get_name(model_name, "concat"))([x, residual])

    return x


def dense_block(tensor,
                units,
                activation="elu",
                model_name="rnn"):
    tensor = BatchNormalization(axis=-1, name=_get_name(model_name, "bn"))(tensor)
    tensor = Dense(units,
                   use_bias=False,
                   kernel_regularizer=regularizers.l2(0.01),
                   name=_get_name(model_name, "dense"))(tensor)
    tensor = Activation(activation, name=_get_name(model_name, "dense"))(tensor)
    # tensor = Dropout(0.5, name=_get_name(model_name, "dropout"))(tensor)
    return tensor


def build_model(x_kin_shape,
                x_pid_shape,
                activation="elu",
                name="rnn"):
    x_kin = Input(x_kin_shape,
                  name=_get_name(name, "input"))
    x_pid = Input(x_pid_shape,
                  name=_get_name(name, "input"))
    x_len = Input(batch_shape=(None, 1),
                  dtype=tf.int32,
                  name=_get_name(name, "input"))

    h_kin = conv1d_block(x_kin, filters=256, activation=activation, first=True, model_name=name)

    h_pid = Embedding(input_dim=8, output_dim=64, name=_get_name(name, "embedding"))(x_pid)
    h_pid = conv1d_block(h_pid, filters=128, activation=activation, first=True, model_name=name)

    h = Concatenate(axis=-1, name=_get_name(name, "concat"))([h_kin, h_pid])

    h = GRU(units=512,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.01),
            name=_get_name(name, "gru"))(h)
    # h = Lambda(gather, name=_get_name(name, "gather"))([h, x_len])

    h = Gather(name=_get_name(name, "gather"))([h, x_len])

    # h = dense_block(h,)
    h = dense_block(h, 256, activation=activation, model_name=name)
    h = dense_block(h, 64, activation=activation, model_name=name)
    h = dense_block(h, 32, activation=activation, model_name=name)
    # h = Dropout(0.5, name=_get_name(name, "dropout"))(h)
    logits = Dense(units=2, name=_get_name(name, "dense"))(h)
    y = Activation("softmax", name=_get_name(name, "softmax"))(logits)

    model = Model(
        inputs=[x_kin, x_pid, x_len],
        outputs=y,
        name=name)

    return model


def get_custom_objects():
    from tensorflow.python.ops.init_ops import glorot_uniform_initializer
    from keras4hep.metrics import roc_auc

    custom_objects = {
        "Gather": Gather,
        "tf": tf,
        "GlorotUniform": glorot_uniform_initializer,
        # "GlorotUniform": keras.initializers.glorot_uniform,
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
