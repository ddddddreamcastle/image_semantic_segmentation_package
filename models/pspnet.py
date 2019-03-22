import torch.nn as nn
from backbone import get_backbone

"""
    Reference:
        Zhao, Hengshuang, et al. "Pyramid scene parsing network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
"""

class PSPCore(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPCore, self).__init__()

        self.branch_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.branch_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.branch_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.branch_6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        tail_channel = in_channels//4
        self.tail = nn.Sequential(
            nn.Conv2d(in_channels*2, tail_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(tail_channel),
            nn.ReLU(True),
            nn.Dropout(0.1, True),
            nn.Conv2d(tail_channel, out_channels, kernel_size=1, stride=1)
        )






class PSPNet(nn.Module):
    def __init__(self, nbr_classes, deep_supervision=True, backbone='resnet50'):
        super(PSPNet, self).__init__()
        self.backbone = get_backbone(backbone, pretrained=True)
        self.psp_core = PSPCore(in_channels=2048, out_channels=nbr_classes)

def get_pspnet(backone='resnet50', pretrained=True):
    psp = PSPNet(150)
    return psp

if __name__ == '__main__':
    psp = get_pspnet()
    for name, f in psp.backbone.named_parameters():
        print(name)


