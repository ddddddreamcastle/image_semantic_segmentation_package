from torch.nn.modules.loss import _Loss

class SegmentationLoss(_Loss):
    def __init__(self, losses, weights):
        super(SegmentationLoss, self).__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, inputs, target):
        

