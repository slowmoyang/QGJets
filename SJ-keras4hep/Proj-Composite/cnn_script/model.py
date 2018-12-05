from collections import defaultdict

import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers

_OP_COUNT = defaultdict(lambda: -1)


def _get_name(model_name, op_name):
    _OP_COUNT[op_name] += 1
    return "{}_{}_{}".format(model_name, op_name, _OP_COUNT[op_name])


def conv_block(tensor,
               filters,
               kernel_size=5,
               strides=(1, 1),
               activation="relu",
               padding="valid",
               order="cab",
               model_name="cnn"):

    for op_name in order:
        if op_name == "c":
            tensor = Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_regularizer=regularizers.l2(0.01),
                name=_get_name(model_name, "conv2d"))(tensor)
        elif op_name == "a":
            if activation in ["LeakyReLU", "leaky_relu"]:
                tensor = layers.LeakyReLU(
                    name=_get_name(model_name, "leaky_relu"))(tensor)
            else:
                tensor = Activation(
                    activation,
                    name=_get_name(model_name, activation))(tensor)
        elif op_name == "b":
            tensor = BatchNormalization(
                axis=1,
                name=_get_name(model_name, "batch_norm"))(tensor)
        else:
            raise ValueError(
                "Expected one of ['conv2d', 'activation', 'batch_norm'], but got {}".format(op_name))

    return tensor


def build_model(x_shape,
                  filters_list=[128, 256, 64],
                  activation="relu",
                  top="proj",
                  kernel_size=5,
                  padding="valid",
                  name="cnn",
                  order="bca"):

    x = Input(x_shape, name=_get_name(name, "input"))

    if not order.startswith("c"):
        h = Conv2D(
            filters=2,
            kernel_size=5,
            kernel_regularizer=regularizers.l2(0.01),
            name=_get_name(name, "conv2d"))(x)
    else:
        h = x 

    for idx, filters in enumerate(filters_list):
        strides = 1 if idx % 2 == 0 else 2
        h = conv_block(tensor=h,
                       filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       activation=activation,
                       order=order,
                       padding=padding,
                       model_name=name)

    if top == "dense":
        h = Flatten(name=_get_name(name, "flatten"))(h)
        logits = Dense(filters=2,
                       kernel_regularizer=regularizers.l2(0.01),
                       name=_get_name(name, "dense"))(h)
    elif top == "gap":
        h = GlobalAveragePooling2D(name=_get_name(name, "gap2d"))(h)
        logits = Dense(2, name=_get_name(name, "dense"))(h)
    elif top == "proj":
        h = Conv2D(filters=2,
                   kernel_size=1,
                   strides=1,
                   kernel_regularizer=regularizers.l2(0.01),
                   name=_get_name(name, "conv2d"))(h)
        logits = GlobalAveragePooling2D(name=_get_name(name, "gap2d"))(h)
    else:
        raise NotImplementedError

    y_pred = Activation("softmax", name=_get_name(name, "softmax"))(logits)

    inputs = [x]
    outputs = [y_pred]

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
    from keras4hep.projects.qgjets.utils import get_dataset_paths
    from dataset import get_data_iter

    paths = get_dataset_paths(min_pt=100)
    data_iter = get_data_iter(paths["training"])

    batch = data_iter.next()
    x_shape = data_iter.get_shape("x", batch_shape=False)

    model = build_model(x_shape, kernel_size=7, name="VanillaConvNet")


    logits = model.predict_on_batch(batch.x)
    print("logits: {}".format(logits.shape))

    model.summary()


if __name__ == "__main__":
    _test()
