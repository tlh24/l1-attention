import math
from torch import nn
from torch.autograd import Function
import torch
import pdb

import l1attn_drv_cpp

torch.manual_seed(42)


class L1AttnFunction(Function):
    @staticmethod
    def forward(ctx, q, k):
        # bs, n_ctx, n_heads, width = q.shape
        q = q.contiguous(); 
        k = k.contiguous();
        attn,c = l1attn_cpp.forward(q, k)
        ctx.save_for_backward(q, k, c)

        return attn

    @staticmethod
    def backward(ctx, d_attn):
        d_q, d_k = l1attn_cpp.backward(d_attn, *ctx.saved_variables)
        return d_q, d_k


class L1Attn(nn.Module):
    def __init__(self):
        super(L1Attn, self).__init__()

    def forward(self, q, k):
        return L1AttnFunction.apply(q, k)
