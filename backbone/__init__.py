from .resnet import get_resnet

def get_backbone(name, **kwargs):
    models = {
        'resnet50': get_resnet(50)
    }
    return models[name](**kwargs)