import tensorflow as tf
import keras.backend as K

"""
In scikit-learn,
    y_true
    y_score



If you want to change the additional args of fn,
then use curring.
"""

def roc_auc_score(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_true, y_pred)

    # find all variables created for this metric
    metric_vars = [each for each in tf.local_variables() if 'auc_roc' in each.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    K.get_session().run(tf.local_variables_initializer())

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
