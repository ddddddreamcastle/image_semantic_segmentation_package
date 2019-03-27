from torch.nn.modules.loss import _Loss

class SegmentationLoss(_Loss):
    def __init__(self, losses, weights):
        super(SegmentationLoss, self).__init__()
        if len(losses) != len(weights):
            raise RuntimeError('the length of `losses` and `weights` have to be equal')
        self.losses = losses
        self.weights = weights

    def forward(self, inputs, target):
        if not isinstance(inputs, tuple):
            raise RuntimeError('`inputs` has to be of type tuple')
        if not isinstance(target, tuple):
            raise RuntimeError('`target` has to be of type tuple')
        if len(inputs) != len(self.losses):
            raise RuntimeError('the length of `losses` and `inputs` have to be equal')
        if len(target) != 1 and len(target) != len(inputs):
            raise RuntimeError('the length of `target` has to be 1 or equals the length of `inputs` ')

        loss = 0
        for idx, pred in enumerate(inputs):
            loss += self.weights[idx] * self.losses[idx](pred, target)
        return loss


