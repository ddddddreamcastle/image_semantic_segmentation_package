from .pspnet import get_pspnet
from .encnet import get_encnet
from .deeplabv3 import get_deeplabv3
def get_model(name, kwargs):
    models = {
        'pspnet': get_pspnet,
        'encnet': get_encnet,
        'deeplabv3': get_deeplabv3
    }
    return models[name](**kwargs)