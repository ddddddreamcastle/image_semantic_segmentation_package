from .pspnet import get_pspnet
from .encnet import get_encnet
def get_model(name, kwargs):
    models = {
        'pspnet': get_pspnet,
        'encnet': get_encnet
    }
    return models[name](**kwargs)