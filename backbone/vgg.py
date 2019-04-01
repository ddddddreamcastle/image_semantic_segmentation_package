from torch import nn

class VGG(nn.Module):
    def __init__(self, nbr_classes=1000, nbr_layers=16):
        super(VGG, self).__init__()