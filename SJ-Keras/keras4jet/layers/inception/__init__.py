"""
(1) arXiv:1409.4842 [cs.CV]
    - Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
      Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
    - Going Deeper with Convolutions

(2) arXiv:1512.00567 [cs.CV]
    - Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
      Zbigniew Wojna
    - Rethinking the Inception Architecture for Computer Vision

(3) arXiv:1602.07261 [cs.CV]
    - Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
    - Inception-v4, Inception-ResNet and the Impact of Residual Connections on
      Learning
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Inception v2 ~ v4
from .inception import inception_a
from .inception import inception_b
from .inception import inception_c

# Inception
from .reduction import reduction_a
from .reduction import reduction_b

# Inception
from .inception_resnet import inception_resnet_a
from .inception_resnet import inception_resnet_b
from .inception_resnet import inception_resnet_c
