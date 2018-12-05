import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate

from keras4hep.projects.qgjets.utils import get_dataset_paths

from cnn_script.model import get_custom_objects as get_cnn_custom_objects
from rnn_script.model import get_custom_objects as get_rnn_custom_objects


def main():
    cnn_path = "/store/slowmoyang/QGJets/SJ-keras4hep/Dev-Composite/VanillaConvNet_epoch-67_loss-0.4987_acc-0.7659_auc-0.8422.hdf5"
    rnn_path = "/store/slowmoyang/QGJets/Keras/Dev-Composite/RNNGatherEmbedding_weights_epoch-121_loss-0.4963_acc-0.7658_auc-0.8431.hdf5"

    print(cnn_path)
    print(rnn_path)

    cnn_custom_objects = get_cnn_custom_objects()
    rnn_custom_objects = get_rnn_custom_objects()

    cnn = load_model(cnn_path, custom_objects=cnn_custom_objects)
    rnn = load_model(rnn_path, custom_objects=rnn_custom_objects)

    ######################################
    # Build
    ######################################
    inputs = cnn.inputs + rnn.inputs

    cnn_last_hidden = cnn.get_layer("cnn_conv2d_3").output
    rnn_last_hidden = rnn.get_layer("rnn_dense_5").output

    cnn_flatten = Flatten()(cnn_last_hidden)

    joint = Concatenate(axis=-1)([cnn_flatten, rnn_last_hidden])

    logits = Dense(2)(joint)

    y_pred = Softmax()(logits)

    model = Model(inputs=inputs, outputs=y_pred)

    model.summary()

    ################################################
    # Freeze
    ##################################################
    for each in model.layers:
        if each.name.startswith("cnn") or each.name.startswith("rnn"):
            each.trainable = False

    for each in model.layers:
        print("{}: {}".format(each.name, each.trainable))



if __name__ == "__main__":
    main()
