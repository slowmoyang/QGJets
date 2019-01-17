from collections import defaultdict

import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers

from keras4hep.layers import MultiHeadSelfAttention
from keras4hep.layers import LayerNormalization

_OP_COUNT = defaultdict(lambda: -1)

def _get_name(model_name, op_name):
    _OP_COUNT[op_name] += 1
    return "{}_{}_{}".format(model_name, op_name, _OP_COUNT[op_name])


def build_classifier(x_shape,
                rnn="gru",
                activation="elu",
                name="rnn"):
    def get_name(op_name):
        return _get_name(name, op_name)

    x = Input(x_shape, name=get_name("input"))

    h = x
    for output_dim in [128, 128, 128, 128, 128]:
        residual = h
        h = MultiHeadSelfAttention(output_dim=output_dim, num_heads=8)(x)
        h = LayerNormalization()(h)
        h = TimeDistributed(Dense(units=64, activation="elu", use_bias=False))(h)
        h = Concatenate()([h, residual])

    h = GlobalAveragePooling1D()(h)
    logits = Dense(units=2, name=get_name("dense"))(h)
    y = Softmax(name=get_name("softmax"))(logits)

    inputs = [x]
    outputs = [y]

    model = Model(inputs=inputs, outputs=outputs, name=name)

    return model



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

    model = build_classifier(x_shape, name="RNN")
    model.summary()

#     y_score = model.predict_on_batch([batch.x])
#     print("y_score: {}".format(y_score.shape))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.WARN)
    _test()
