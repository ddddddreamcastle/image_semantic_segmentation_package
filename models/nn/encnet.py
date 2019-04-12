import torch.nn as nn
from datasets import datasets
import torch
from backbone import get_backbone
from .encoding import scaled_l2, aggregate

class EncCore(nn.Module):
    def __init__(self, in_channels, out_channels, up_method, se_loss):
        super(EncCore, self).__init__()
        self.top = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.encoding = nn.Sequential(
            nn.Conv2d(512, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(512),

        )


    def forward(self, x):
        pass

class EncNet(nn.Module):
    def __init__(self, nbr_classes, deep_supervision=True, backbone='resnet50', se_loss=True, **kwargs):
        super(EncNet, self).__init__()
        self.backbone = get_backbone(backbone, backbone_pretrained=True, **kwargs)
        self.core = EncCore(in_channels=2048, out_channels=nbr_classes, up_method=self.up_method, se_loss=se_loss)

    def forward(self, x):
        pass

def get_encnet(backbone='resnet50', model_pretrained=True, supervision=True,
               model_pretrain_path=None, dataset='ade20k', **kwargs):
    nbr_classes = datasets[dataset].NBR_CLASSES
    psp = EncNet(nbr_classes, supervision, backbone, **kwargs)
    if model_pretrained:
        psp.load_state_dict(torch.load(model_pretrain_path)['state_dict'], strict=False)
        print("model weights are loaded successfully")
    return psp