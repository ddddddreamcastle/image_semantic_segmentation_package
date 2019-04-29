from .resnet import get_resnet
from .xception import get_xception

def get_backbone(name, **kwargs):
    models = {
        'resnet50':  get_resnet(50),
        'xception': get_xception()
    }
    return models[name](**kwargs)