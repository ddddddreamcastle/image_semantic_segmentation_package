import torch.nn as nn
from backbone import get_backbone
from torch.nn import functional as F
import torch

"""
    Reference:
        Zhao, Hengshuang, et al. "Pyramid scene parsing network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
"""

class PSPCore(nn.Module):
    def __init__(self, in_channels, out_channels, up_method):
        super(PSPCore, self).__init__()

        branch_channels = in_channels // 4

        self.branch_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True)
        )

        self.branch_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True)
        )

        self.branch_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True)
        )

        self.branch_6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(in_channels*2, branch_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(branch_channels, out_channels, kernel_size=1, stride=1)
        )

        self.up_method = up_method

    def forward(self, x):
        _, _, h, w = x.size()
        branch1 = F.interpolate(self.branch_1(x), (h, w), **self.up_method)
        branch2 = F.interpolate(self.branch_2(x), (h, w), **self.up_method)
        branch3 = F.interpolate(self.branch_3(x), (h, w), **self.up_method)
        branch6 = F.interpolate(self.branch_6(x), (h, w), **self.up_method)
        x = torch.cat((x, branch1, branch2, branch3, branch6), 1)
        return self.tail(x)


class PSPNet(nn.Module):
    def __init__(self, nbr_classes, deep_supervision=True, backbone='resnet50', **kwargs):
        super(PSPNet, self).__init__()
        self.up_method = {'mode': 'bilinear', 'align_corners': True}
        self.deep_supervision = deep_supervision
        self.backbone = get_backbone(backbone, pretrained=True, aux=True)
        self.psp_core = PSPCore(in_channels=2048, out_channels=nbr_classes, up_method=self.up_method)
        if deep_supervision:
            self.aux_branch = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(256, nbr_classes, kernel_size=1, stride=1)
            )

    def forward(self, x):
        _, _, h, w = x.size()
        x, aux = self.backbone(x)
        x = self.psp_core(x)
        x = F.interpolate(x, (h, w), **self.up_method)
        if self.deep_supervision:
            aux = self.aux_branch(aux)
            aux = F.interpolate(aux, (h, w), **self.up_method)
            return x, aux
        return x

def get_pspnet(backbone='resnet50', model_pretrained=True, supervision=True,
               model_pretrain_path=None, **kwargs):

    psp = PSPNet(150, supervision, backbone, **kwargs)
    if model_pretrained:
        psp.load_state_dict(torch.load(model_pretrain_path), strict=False)
    return psp

if __name__ == '__main__':
    psp = get_pspnet()
    for name, f in psp.backbone.named_parameters():
        print(name)