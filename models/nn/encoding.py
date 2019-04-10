import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

class ScaledL2(Function):

    @staticmethod
    def forward(ctx, X, C, S):
        SL = S.view(1, 1, C.size(0)) * (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) -
             C.unsqueeze(0).unsqueeze(0)).pow(2).sum(3)
        ctx.save_for_backward(X, C, S, SL)
        return SL

    @staticmethod
    def backward(ctx, grad_outputs):
        X, C, S, SL = ctx.saved_variables