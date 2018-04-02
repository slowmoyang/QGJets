from keras.layers.core import Lambda
from keras.models import Sequential
import keras.backend as K
import keras
import tensorflow as tf
import numpy as np
import cv2


def target_category_loss(x, category_index, num_classes):
    return tf.multiply(x, K.one_hot([category_index], num_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape



def get_grad_cam(model, image_batch, layer_name=None, num_classes=2, mode="positive"):
    """
    Add Grad-CAM graph
    """
    prediction = model.predict(image_batch)[0]
    predicted_class = np.argmax(prediction)

    grad_cam_model = Sequential()
    grad_cam_model.add(model)

    target_layer = lambda x: target_category_loss(x, predicted_class, num_classes) # currying

    grad_cam_model.add(
        Lambda(
            function=target_layer,
            output_shape=target_category_loss_output_shape
        )
    )

    # the score for class c, y^c
    score = K.sum(model.layers[-1].output)

    # Get the output feature maps of last convolutional layer
    # feature maps A^k of a convolutional layer
    #
    # last_fmaps: feature maps of last convolutional layers
    if layer_name is None:
        for layer in model.layers[::-1]:
            if isinstance(layer, keras.layers.Conv2D):
                last_fmaps = layer.output
                break
    else:
        last_fmaps = model.get_layer(layer_name).output

    # K.gradients() returns a gradients tensor with NCHW format
    # [1, C, H, W]
    grads = K.gradients(score, last_fmaps)[0]
    grads = K.l2_normalize(grads)

    gradient_fn = K.function(
        inputs=[grad_cam_model.layers[0].input, K.learning_phase()],
        outputs=[last_fmaps, grads])


    """
    Run the graph
    """
    last_fmaps_np, grads_np = gradient_fn([image_batch, True])

    # pick first one from batch
    last_fmaps_np = last_fmaps_np[0]
    grads_np = grads_np[0]

    # the neuron importance weights \alpha^c_k
    weights = np.mean(grads_np, axis = (1, 2)) # GAP
    weights = weights.reshape(-1, 1, 1) # (c, 1, 1)

    # We perform a weighted combination of forward activation maps
    grad_cam = np.sum(weights*last_fmaps_np, axis=0)

    # and follow it by a ReLU.
    grad_cam = np.maximum(grad_cam, 0) # ReLU


    """ Visualization """
    # cv2.resize(src, dsize)
    # src: input image
    # dsize: output image size
    grad_cam_resize = cv2.resize(src=grad_cam, dsize=image_batch.shape[2:])
    return grad_cam, grad_cam_resize
