import torch.nn as nn
from backbone import get_backbone

"""
    Reference:
        Zhao, Hengshuang, et al. "Pyramid scene parsing network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
"""

class PSPNet(nn.Module):
    def __init__(self, nbr_classes, deep_supervision=True, backbone='resnet50'):
        super(PSPNet, self).__init__()
        self.backbone = get_backbone(backbone, pretrained=True)


def get_pspnet(backone='resnet50', pretrained=True):
    psp = PSPNet(150)
    return psp

if __name__ == '__main__':
    psp = get_pspnet()
    for name, f in psp.backbone.named_parameters():
        print(name, f.data)
        break


