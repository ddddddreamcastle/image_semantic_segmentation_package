from .pspnet import get_pspnet

def get_model(name, **kwargs):
    models = {
        'pspnet': get_pspnet
    }
    return models[name](**kwargs)