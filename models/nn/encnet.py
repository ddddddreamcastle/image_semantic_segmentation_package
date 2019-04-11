import torch.nn as nn
from datasets import datasets
import torch
from .encoding import scaled_l2, aggregate

class EncCore(nn.Module):
    def __init__(self, in_channels, out_channels, up_method):
        super(EncCore, self).__init__()

    def forward(self, *input):
        pass

class EncNet(nn.Module):
    def __init__(self, nbr_classes, deep_supervision=True, backbone='resnet50', **kwargs):
        super(EncNet, self).__init__()

    def forward(self, *input):
        pass

def get_encnet(backbone='resnet50', model_pretrained=True, supervision=True,
               model_pretrain_path=None, dataset='ade20k', **kwargs):
    nbr_classes = datasets[dataset].NBR_CLASSES
    psp = EncNet(nbr_classes, supervision, backbone, **kwargs)
    if model_pretrained:
        psp.load_state_dict(torch.load(model_pretrain_path)['state_dict'], strict=False)
        print("model weights are loaded successfully")
    return psp