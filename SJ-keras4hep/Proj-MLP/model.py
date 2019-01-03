from collections import defaultdict

from  tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers

from dataset import BDTVarSet

_OP_COUNT = defaultdict(lambda: -1)

def _get_name(model_name, op_name):
    _OP_COUNT[op_name] += 1
    return "{}_{}_{}".format(model_name, op_name, _OP_COUNT[op_name])


def dense_block(tensor,
                units,
                activation="elu",
                model_name="rnn"):

    def get_name(op_name):
        return _get_name(model_name, op_name)

    tensor = Dense(units,
                   use_bias=False,
#                    kernel_regularizer=regularizers.l2(0.01),
                   name=get_name("dense"))(tensor)

    tensor = BatchNormalization(
        axis=-1,
        momentum=0.9,
        name=get_name("batch_normalization"))(tensor)


    tensor = Activation(
        activation,
        name=get_name(activation))(tensor)

    return tensor


def build_model(x_shape,
                name="mlp",
                activation="elu"):
    def get_name(op_name):
        return _get_name(name, op_name)

    x = Input(x_shape, name=get_name("input"))

    h = dense_block(x, 64, activation=activation, model_name=name)
    h = dense_block(h, 256, activation=activation, model_name=name)
    h = dense_block(h, 256, activation=activation, model_name=name)
    h = dense_block(h, 256, activation=activation, model_name=name)
    h = dense_block(h, 256, activation=activation, model_name=name)
    h = dense_block(h, 64, activation=activation, model_name=name)
    logits = Dense(units=2, name=get_name("dense"))(h)
    y = Softmax(name=get_name("softmax"))(logits)

    model = Model(
        inputs=x,
        outputs=y,
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
    data_iter = get_data_iter(path, full_info=False)

    batch = data_iter.next()
    x_shape = data_iter.get_shape("x", batch_shape=False)

    model = build_model(x_shape)
    model.summary()

    logits = model.predict_on_batch(batch.x)
    print("logits: {}".format(logits.shape))

if __name__ == "__main__":
    _test()


