import torch
import numpy as np

class SegmentationErrorMeter(object):

    def __init__(self, metrics, nbr_classes):
        self.metrics = metrics
        self.total_correct = 0
        self.total_label = 0
        self.total_inter = 0
        self.total_union = 0
        self.nbr_classes = nbr_classes


    def reset(self):
        self.total_correct = 0
        self.total_labeled = 0
        self.total_inter = 0
        self.total_union = 0

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy().astype('int32') + 1
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy().astype('int32') + 1
        correct, labeled = batch_pixel_accuracy(output, target)
        self.total_correct += correct
        self.total_labeled += target

def batch_pixel_accuracy(output, target):
    output = np.argmax(output, axis=1)
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((output == target) * (target > 0))
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nbr_classes):
    output = np.argmax(output, axis=1)

