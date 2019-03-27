import torch

class SegmentationErrorMeter(object):

    def __init__(self, metrics):
        self.metrics = metrics
        self.total_correct = 0
        self.total_label = 0
        self.total_inter = 0
        self.total_union = 0

    def reset(self):
        self.total_correct = 0
        self.total_label = 0
        self.total_inter = 0
        self.total_union = 0

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()

    