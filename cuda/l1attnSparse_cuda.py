import math
from torch import nn
from torch.autograd import Function
import torch
import l1attnSparse_cuda

class L1AttnSparseFn(Function):
    @staticmethod
    def forward(ctx, q, k):
        q = q.contiguous(); 
        k = k.contiguous();
        attn,c = l1attn_cuda.forward(q, k)
        ctx.save_for_backward(q, k, c)
        return attn

    @staticmethod
    def backward(ctx, d_attn):
        d_q, d_k = l1attn_cuda.backward(d_attn, *ctx.saved_variables)
        return d_q, d_k


class L1AttnSparse(nn.Module):
    def __init__(self):
        super(L1AttnSparse, self).__init__()

    def forward(self, q, k):
        return L1AttnSparseFn.apply(q, k)
