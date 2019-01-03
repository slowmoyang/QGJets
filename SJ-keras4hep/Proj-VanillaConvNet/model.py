from collections import defaultdict

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


_OP_COUNT = defaultdict(lambda: -1)


def _get_name(model_name, op_name):
    _OP_COUNT[op_name] += 1
    return "{}_{}_{}".format(model_name, op_name, _OP_COUNT[op_name])


def conv_block(tensor,
               filters,
               model_name,
               kernel_size=5,
               strides=(1, 1),
               activation="relu",
               padding="valid"):

    def get_name(op_name):
        return _get_name(model_name, op_name)

    tensor = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    name=get_name("conv2d"))(tensor)

    tensor = Activation(activation, name=get_name(activation))(tensor)

    tensor = BatchNormalization(axis=1, name=get_name("batch_norm"))(tensor)
    return tensor


def build_model(x_shape,
                filters_list=[16, 64, 32],
                activation="relu",
                top="proj",
                name="cnn"):

    def get_name(op_name):
        return _get_name(name, op_name)


    x = Input(x_shape)

    h = x
    for filters in filters_list:
        h = conv_block(h, filters, activation=activation, model_name=name)

    if top == "dense":
        h = Flatten(name=get_name("flatten"))(h)
        logits = Dense(2, name=get_name("dense"))(h)
    elif top == "dense2":
        h = Flatten(name=get_name("flatten"))(h)
        h = Dense(32, name=get_name("dense"))(h)
        logits = Dense(2, name=get_name("dense"))(h)
    elif top == "proj":
        h = Conv2D(filters=2, kernel_size=1, strides=1, name=get_name("conv2d"))(h)
        logits = GlobalAveragePooling2D(name=get_name("gap2d"))(h)
    elif top == "gap":
        h = GlobalAveragePooling2D(name=get_name("gap2d"))(h)
        logits = Dense(2, name=get_name("dense"))(h)
    else:
        raise NotImplementedError

    y_score = Softmax(name=get_name("softmax"))(logits)

    model = Model(inputs=x,
                  outputs=y_score,
                  name=name)

    return model


def _test():
    from dataset import get_data_iter
    from keras4hep.projects.qgjets.utils import get_dataset_paths
    path = get_dataset_paths(min_pt=100)["training"]
    data_iter = get_data_iter(path)


    batch = data_iter.next()
    x_shape = data_iter.get_shape("x", batch_shape=False)

    model = build_model(x_shape)

    logits = model.predict_on_batch(batch.x)
    print("logits: {}".format(logits.shape))


if __name__ == "__main__":
    _test()
