import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *


def conv1d_block(x, filters, kernel_size=1, strides=1, activation="elu"):
    x = Conv1D(filters, kernel_size, strides=strides)(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis=-1)(x)
    return x


def dense_block(x, units, activation="elu"):
    x = Dense(units)(x)
    x = Activation(activation)(x)
    x = BatchNormalization(axis=-1)(x)
    return x


def gather(args): 
    """ref. https://stackoverflow.com/questions/41267829/retrieving-last-value-of-lstm-sequence-in-tensorflow
    Alex's solution
    args = (padded_seq, seq_len)
    """
    # Lambda layer can get only one argument
    assert len(args) == 2
    seq, idx = args

    # Keras standardize input shape
    # (batch_size, ) --> (batch_size, 1)
    idx = tf.reshape(idx, shape=(-1, ))

    batch_range = tf.range(tf.shape(seq)[0])
    indices = tf.stack([batch_range, idx], axis=1)
    return tf.gather_nd(seq, indices)

    


def build_a_model(x_shape):
    x = Input(x_shape)
    x_len = Input(batch_shape=(None,1), dtype=tf.int32)

    # h = SpatialDropout1D(rate=0.8)(x)
    # h = Bidirectional(LSTM(units=512, return_sequences=True))(h)
    h = LSTM(units=512, return_sequences=True)(x)
    h = Lambda(gather)([h, x_len])
    h = dense_block(h, 128)
    h = dense_block(h, 64)
    h = dense_block(h, 16)
    logits = Dense(units=2)(h)
    y = Activation("softmax")(logits)

    model = Model(
        inputs=[x, x_len],
        outputs=y)

    return model

def _test():
    from keras4hep.data import DataIterator
    from dataset import JetSeqSet

    path = "/store/slowmoyang/QGJets/dijet_100_110/dijet_100_110_test.root"
    prep_path = "./logs/dijet_100_110_training.npz"
    dset = JetSeqSet(path, extra=["pt", "eta"],
                     seq_maxlen={"x": 50})
    data_iter = DataIterator(dset, batch_size=128)

    batch = data_iter.next()
    x_shape = data_iter.get_shape("x", batch_shape=False)
    x_len_shape = data_iter.get_shape("x_len", batch_shape=False)
    print(x_shape)
    print(x_len_shape)

    print(batch.x_len.shape)

    model = build_a_model(x_shape)

    y_score = model.predict_on_batch([batch.x, batch.x_len])
    print("y_score: {}".format(y_score.shape))


if __name__ == "__main__":
    _test()
