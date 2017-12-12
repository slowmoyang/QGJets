from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import models.image.vggnet
# import models.resnet
# import models.densenet
# import models.seresnet
# import models.sedensnet

# import models.capsnet

__all__ = []

def build_a_model(model_name, *args, **kargs):
    return getattr(models, model_name).build_a_model(*args, **kargs)
