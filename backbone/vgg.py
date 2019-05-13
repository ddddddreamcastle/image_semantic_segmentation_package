from torch import nn
from torchviz import make_dot
import torch
from models.components.norm import get_norm

cfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, nbr_classes=1000, nbr_layers=16, batch_norm=True, dilation=False, norm='bn'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[nbr_layers], batch_norm=batch_norm, dilation=dilation, norm=norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, nbr_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, batch_norm=True, dilation=True, norm='bn'):
        layers = []
        in_channels = 3
        multi_grid = [1,2,4,8]
        i = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                i = 0
            else:
                if not dilation:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=multi_grid[i], dilation=multi_grid[i])
                    i += 1
                if batch_norm:
                    layers += [conv2d, get_norm(norm, channels=v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

def get_vgg(nbr_layers=16, batch_norm=True, dilation=False):
    def build_net(backbone_pretrained_path='./weights/vgg16.pth', nbr_classes=1000,
                  backbone_pretrained=True, norm='bn', **kwargs):
        model = VGG(nbr_classes, nbr_layers=nbr_layers, batch_norm=batch_norm, dilation=dilation, norm=norm)
        if backbone_pretrained:
            model.load_state_dict(torch.load(backbone_pretrained_path), strict=False)
            print('vgg weights are loaded successfully')
        return model
    return build_net

if __name__ == '__main__':
    vgg = VGG(nbr_classes=1000, nbr_layers=16, batch_norm=True, dilation=False)
    g = make_dot(vgg(torch.rand(16, 3, 224, 224)), params=dict(vgg.named_parameters()))
    g.render('vgg16')
    params = list(vgg.parameters())
    k = 0
    for i in params:
        l = 1
        print("layer architecture：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("the number of parameters：" + str(l))
        k = k + l
    print("the total of parameters：" + str(k))