import math
from torch import nn
from torch.autograd import Function
import torch
import pdb

import l1attn_drv_cpp

torch.manual_seed(42)


class L1AttnFn(Function):
    @staticmethod
    def forward(ctx, q, k):
        # bs, n_ctx, n_heads, width = q.shape
        q = q.contiguous(); 
        k = k.contiguous();
        attn = l1attn_drv_cpp.forward(q, k)
        ctx.save_for_backward(q, k)
        return attn[0] # unpack

    @staticmethod
    def backward(ctx, d_attn):
        q, k = ctx.saved_variables[:2]
        d_q, d_k = l1attn_drv_cpp.backward(d_attn, q, k)
        return d_q, d_k


class L1Attn(nn.Module):
    def __init__(self):
        super(L1Attn, self).__init__()

    def forward(self, q, k):
        return L1AttnFn.apply(q, k)
