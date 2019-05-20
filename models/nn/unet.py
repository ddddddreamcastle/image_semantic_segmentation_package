import torch.nn as nn
from backbone import get_backbone
from torch.nn import functional as F
import torch
from datasets import datasets
from models.components.norm import get_norm
from torchviz import make_dot

"""
    Reference:
        Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
"""

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bn', up_sample=False):
        super(Up, self).__init__()
        if up_sample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            get_norm(name=norm, channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            get_norm(name=norm, channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        _, _, h2, w2 = x2.size()
        _, _, h1, w1 = x1.size()

        diffH = h2 - h1
        diffW = w2 - w1

        x1 = F.pad(x1, (diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2))
        x = torch.cat([x1, x2], dim=1)

        x = self.conv(x)

        return x

class UNetCore(nn.Module):
    def __init__(self, out_channels, norm='bn', up_sample=False):
        super(UNetCore, self).__init__()

        self.up1 = Up(512, 512, up_sample=up_sample, norm=norm)
        self.up2 = Up(512, 256, up_sample=up_sample, norm=norm)
        self.up3 = Up(256, 128, up_sample=up_sample, norm=norm)
        self.up4 = Up(128, 64, up_sample=up_sample, norm=norm)
        self.up5 = Up(64, 32, up_sample=up_sample, norm=norm)

        self.conv = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connections):
        _, _, h, w = x.size()
        x = self.up1(x, skip_connections[0])
        x = self.up2(x, skip_connections[1])
        x = self.up3(x, skip_connections[2])
        x = self.up4(x, skip_connections[3])
        x = self.up5(x, skip_connections[4])
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, nbr_classes, backbone='vgg16', norm='bn', **kwargs):
        super(UNet, self).__init__()
        self.nbr_classes = nbr_classes
        up_sample = False
        self.backbone = get_backbone(backbone, **kwargs)
        self.core = UNetCore(out_channels=nbr_classes,
                             norm=norm, up_sample=up_sample)

    def get_parameters_as_groups(self, lr, different_lr_in_layers=True):
        parameters = []
        if different_lr_in_layers:
            parameters.append({'params': self.backbone.parameters(), 'lr':lr})
            parameters.append({'params': self.core.parameters(), 'lr':lr*10})
        else:
            parameters.append({'params': self.parameters(), 'lr': lr })
        return parameters

    def forward(self, x):
        x = self.backbone.backbone_forward(x)
        x = self.core(x, self.backbone.skip_connections)
        return x

def get_unet(backbone='vgg16', model_pretrained=True,
               model_pretrain_path=None, dataset='ade20k', norm='bn', **kwargs):
    nbr_classes = datasets[dataset].NBR_CLASSES
    psp = UNet(nbr_classes, backbone, norm=norm, **kwargs)
    if model_pretrained:
        psp.load_state_dict(torch.load(model_pretrain_path)['state_dict'], strict=False)
        print("model weights are loaded successfully")
    return psp

if __name__ == '__main__':
    model = get_unet(backbone='vgg16', model_pretrained=False, backbone_pretrained=False)
    g = make_dot(model(torch.rand(16, 3, 256, 256)), params=dict(model.named_parameters()))
    g.render('unet_vgg16')