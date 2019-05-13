from .resnet import get_resnet
from .xception import get_xception
from .vgg import get_vgg

def get_backbone(name, **kwargs):
    models = {
        'resnet50':  get_resnet(50),
        'resnet101': get_resnet(101),
        'resnet152': get_resnet(152),
        'xception': get_xception(),
        'vgg11': get_vgg(11, False),
        'vgg13': get_vgg(13, False),
        'vgg16': get_vgg(16, False),
        'vgg19': get_vgg(19, False),
        'vgg11bn': get_vgg(11, True),
        'vgg13bn': get_vgg(13, True),
        'vgg16bn': get_vgg(16, True),
        'vgg19bn': get_vgg(19, True)
    }
    return models[name](**kwargs)