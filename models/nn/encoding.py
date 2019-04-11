import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

class scaled_L2(Function):

    @staticmethod
    def forward(ctx, X, C, S):
        SL = S.view(1, 1, C.size(0)) * (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) -
             C.unsqueeze(0).unsqueeze(0)).pow(2).sum(3)
        ctx.save_for_backward(X, C, S, SL)
        return SL

    @staticmethod
    def backward(ctx, grad_SL):
        X, C, S, SL = ctx.saved_variables
        tmp = (2 * grad_SL * S.view(1, 1, C.size(0))).unsqueeze(3) * \
                (X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1)) - C.unsqueeze(0).unsqueeze(0))
        GX = tmp.sum(2)
        GC = tmp.sum(0).sum(0)
        GS = (grad_SL * (SL / S.view(1, 1, C.size(0)))).sum(0).sum(0)
        return GX, GC, GS

class aggregate(Function):

    @staticmethod
    def forward(ctx, A, X, C):
        ctx.save_for_backward(A, X, C)
        E = (A.unsqueeze(3) * (X.unsqueeze(2).expand(X.size(0), X.size(1),
                                                      C.size(0), C.size(1)) - C.unsqueeze(0).unsqueeze(0))).sum(1)
        return E

    @staticmethod
    def backward(ctx, grad_E):




