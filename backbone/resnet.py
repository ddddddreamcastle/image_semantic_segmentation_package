import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F

"""
    Reference:
        [1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        [2] Yu, Fisher , and V. Koltun . "Multi-Scale Context Aggregation by Dilated Convolutions." (2015).
        [3] Yu, Fisher , V. Koltun , and T. Funkhouser . "Dilated Residual Networks." 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) IEEE Computer Society, 2017.
"""

def conv3x3(in_channel, out_channel, stride=1):
    """ 3x3 convolution layer """
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    """ Basic residual block for Dilated Residual Networks (DRN) """
    def __init__(self, in_channel, out_channel, stride=1, dilation=1,
                 downsample=None, residual=True):
        super(BasicBlock, self).__init__()
        self.conv_1_3x3 = nn.Conv2d(in_channel, out_channel, stride, padding=dilation, dilation=dilation)
        self.bn_1 = nn.BatchNorm2d(out_channel)

        self.conv_2_3x3 = nn.Conv2d(out_channel, out_channel, stride, padding=dilation, dilation=dilation)
        self.bn_2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv_1_3x3(x)
        out = F.relu(self.bn_1(out), True)

        out = self.conv_2_3x3(out)
        out = F.relu(self.bn_2(out), True)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.residual:
            out += residual

        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    """ Bottleneck for Dilated Residual Networks (DRN) """
    def __init__(self, in_channel, out_channel, stride=1, dilation=1,
                 downsample=None, residual=True):
        super(Bottleneck, self).__init__()
        # 1x1
        self.conv_1_1x1 = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channel)

        # 3x3
        self.conv_2_3x3 = nn.Conv2d(out_channel, out_channel, 3, stride=stride, padding=dilation, dilation=dilation,
                                    bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channel)

        # 1x1
        self.conv_3_1x1 = nn.Conv2d(out_channel, 4 * out_channel, 1, bias=False)
        self.bn_3 = nn.BatchNorm2d(4 * out_channel)

        self.downsample = downsample
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv_1_1x1(x)
        out = F.relu(self.bn_1(out), True)

        out = self.conv_2_3x3(out)
        out = F.relu(self.bn_2(out), True)

        out = self.conv_3_1x1(out)
        out = F.relu(self.bn_3(out), True)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.residual:
            out += residual

        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, pretrained=False, nbr_classes=1000, is_bottleneck = True, nbr_layers=50):
        super(ResNet, self).__init__()
        self.head = nn.Sequential(
            conv3x3(3, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        net_structures = {50: [3, 4, 6, 3],
                          101: [3, 4, 23, 3],
                          152: [3, 8, 36, 3]}
        residual_block = Bottleneck if is_bottleneck else BasicBlock
        self.block_1 = self._make_layers(residual_block, 128, 64, )



    def _make_layers(self, block, in_channel, out_channel, nbr_blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * block.expansion)
            )
        layers = []
        layers.append(block(in_channel, out_channel, stride=stride, downsample=downsample,
            dilation=1 if dilation == 1 else dilation // 2))

        in_channel = out_channel * block.expansion

        for _ in range(1, nbr_blocks):
            layers.append(block(in_channel, out_channel, stride=stride, dilation=dilation))

        return nn.Sequential(*layers)





