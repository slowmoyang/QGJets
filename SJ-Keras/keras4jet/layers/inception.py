"""
ref. arXiv:1409.4842 [cs.CV]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.utils import conv_utils
import keras.backend as K

if __name__ == "__main__":
    import layer_utils
    from layer_utils import conv_unit
    from layer_utils import factorized_conv
else:
    from . import layer_utils
    from .layer_utils import factorized_conv







def inception(x,
              filters=None,
              kernel_size=None,
              avg_pool=True,
              factorization=None,
              asym=None,
              name=None):
    """
    if factorization:
        if asym:
            name = "Inception-B_v2"
        else:
            name = "Inception-A_v2"
    else:
        name = "Inception_v1"
    """
    channel_axis = layer_utils.get_channel_axis()
    h, w = K.int_shape(x)[2:] if channel_axis == 1 else K.int_shape(x)[1:3]
    assert h == w

    if filters is None:
        in_channels = K.int_shape(x)[channel_axis]

        # unit
        u = int(in_channels/4)
        filters = {
            "branch0_bottleneck": u, # or conv5x5_reduce
            "branch0": int(u*1.5),
            "branch1_bottleneck": int(u*0.75), 
            "branch1": u,
            "branch2": int(u/2),
            "branch3": u}


    if factorization is None:
        if ( h > 20 ):
            factorization = False
            asym = False
        elif ( h >= 12 ) and ( h <= 20): 
            factorization = True
            asym = True
        else: # h < 12
            factorization = True
            asym = True

    if kernel_size is None:
        if factorization:
            if asym:
                k = 7 # Inception-B
            else:
                k = 5 # inception-A
        else:
            k = 5 # Inveption v1

        kernel_size = (k, k)

        

 
    # Branch0 : conv5x5
    x0 = conv_unit(x, filters=filters["branch0_bottleneck"], kernel_size=(1,1), strides=(1,1))

    if factorization:
        x0 = factorized_conv(
            x,
            filters=filters["branch0"],
            kernel_size=kernel_size,
            strides=(1,1),
            asym=asym)
    else:
        x0 = conv_unit(
            x,
            filters=filters["branch0"],
            kernel_size=kernel_size,
            strides=(1,1))
        
    # Branch1 : conv3x3
    x1 = conv_unit(x, filters=filters["branch1_bottleneck"], kernel_size=(1,1), strides=(1,1))
    x1 = conv_unit(x, filters=filters["branch1"], kernel_size=(3,3), strides=(1, 1))

    # Branch2: pool proj
    pool_kargs = {"pool_size": (3,3), "strides": 2}
    if avg_pool:
        x2 = AveragePooling2D(**pool_kargs)(x)
    else:
        x2 = MaxPooling2D(**pool_kargs)(x) # pooling layer
    
    x2 = conv_unit(x, filters=filters["branch2"], kernel_size=(1,1), strides=(1,1)) # projection layer

    # Branch : conv3x3
    x3 = conv_unit(x, filters=filters["branch3"], kernel_size=(1,1))

    # Filter concat
    filters_concat = Concatenate(axis=channel_axis)([x0, x1, x2, x3]) # out_channels = 256

    return filters_concat



# def inception(x, filters=None, kernel_size, avg_pool=True, factorization=True, asym=True, name=None):

def inception_naive():
    pass

def inception_v1(x, filters=None, kernel_size, avg_pool=True, name=None):
    return inception(x, filters, kernel_size, avg_pool, factorization=False, asym=False, name=name)

def inception_a(x, filters=None, kernel_size, avg_pool=True, name=None):
    return inception(x, filters, kernel_size, avg_pool, factorization=True, asym=False, name=name)

def inception_b(x, filters=None, kernel_size, avg_pool=True, name=None):
    return inception(x, filters, kernel_size, avg_pool, factorization=True, asym=True, name)

def reduction_a():
    pass

def reduction_b():
    pass








if __name__ == "__main__":
    from keras.layers import Input
    from keras.models import Model
    from keras.utils import plot_model

    inputs = Input((3, 224, 224))
    conv7x7 = Conv2D(filters=32, kernel_size=(7,7), name="Conv7x7")(inputs)
    inception = inception_a(conv7x7, as_model=True)

    model = Model(inputs=inputs, outputs=inception)
    plot_model(model, to_file="./model_plot/inception.png", show_shapes=True)  
