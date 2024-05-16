import torch
from torch import nn
from torch.autograd import Function
# torch must be imported before extension, o/w shared-object links to c10 etc don't work
import l1attn_cuda_drv

class L1AttnFn(Function):
    @staticmethod
    def forward(ctx, q, k):
        n_heads = q.shape[2]
        assert(n_heads <= 8)
        q = q.contiguous(); 
        k = k.contiguous();
        attn = l1attn_cuda_drv.forward(q, k)
        ctx.save_for_backward(q, k)
        return attn[0]

    @staticmethod
    def backward(ctx, d_attn):
        q, k = ctx.saved_variables[:2]
        n_heads = q.shape[2]
        assert(n_heads <= 8)
        # q & k are bthw & bshw
        # transpose them so memory access across t,s is coalesced
        q = q.transpose(1,3).contiguous() # bthw -> bwht
        k = k.transpose(1,3).contiguous() # bshw -> bwhs
        d_attnq = d_attn.transpose(1,3).transpose(1,2).contiguous() 
                # bsth -> bhts -> bths
        d_attnk = d_attn.transpose(2,3).contiguous() # bsth -> bsht
        d_q, d_k = l1attn_cuda_drv.backward(d_attnq, d_attnk, q, k)
        return d_q, d_k # output has no transpose; writes can be cached.


class L1Attn(nn.Module):
    def __init__(self):
        super(L1Attn, self).__init__()

    def forward(self, q, k):
        return L1AttnFn.apply(q, k)
