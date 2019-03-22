from .resnet import get_resnet

def get_backbone(name, **kwargs):
    models = {
        'resnet': get_resnet
    }
    return models[name](**kwargs)