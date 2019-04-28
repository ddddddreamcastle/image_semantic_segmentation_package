from torch import nn

"""
    Reference:
        [1] https://arxiv.org/pdf/1610.02357.pdf
        [2] Chollet, F.: Xception: Deep learning with depthwise separable convolutions. In: CVPR. (2017)
        [3] Qi, H., Zhang, Z., Xiao, B., Hu, H., Cheng, B., Wei, Y., Dai, J.: Deformable convolutional networks – coco detection and segmentation challenge 2017 entry. ICCV COCO Challenge Workshop (2017)
"""

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 depth_activation=True):
        super(SeparableConv2d, self).__init__()
        self.relu_0 = nn.ReLU(True)

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, bias=bias, groups=in_channels)
        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.relu_1 = nn.ReLU(True)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu_2 = nn.ReLU(True)

        self.depth_activation = depth_activation

    def forward(self, x):
        if not self.depth_activation:
            x = self.relu_0(x)
        x = self.depthwise(x)
        x = self.bn_1(x)
        if self.depth_activation:
            x = self.relu_1(x)
        x = self.pointwise(x)
        x = self.bn_2(x)
        if self.depth_activation:
            x = self.relu_2(x)
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, bias=False,
                 depth_activation=True, grow_first=True):
        super(Block, self).__init__()

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        if grow_first:
            inner_channels = out_channels
        else:
            inner_channels = in_channels

        self.layer_1 = SeparableConv2d(in_channels, inner_channels, 3, stride=1, padding=dilation, dilation=dilation,
                                       bias=False, depth_activation=depth_activation)
        self.layer_2 = SeparableConv2d(inner_channels, out_channels, 3, stride=1, padding=dilation, dilation=dilation,
                                       bias=False, depth_activation=depth_activation)
        self.layer_3 = SeparableConv2d(out_channels,out_channels,3,stride=stride, padding=dilation,dilation=dilation,
                                        bias=False,depth_activation=depth_activation)

    def forward(self, x):
        skip = x
        if self.skip != None:
            skip = self.skip(x)
            skip = self.skip(skip)
        main = self.layer_1(x)
        main = self.layer_2(main)
        main = self.layer_3(main)

        main += skip
        return main

class Xception(nn.Module):
    def __init__(self, nbr_classes=1000, os=8):
        super(Xception, self).__init__()

