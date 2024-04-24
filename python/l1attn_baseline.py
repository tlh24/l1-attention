import math

from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import pdb


class L1AttnFn(Function):
	@staticmethod
	def forward(ctx, q, k):
		bs, n_ctx, n_heads, width = q.shape # shape!
		scale = -1 / math.sqrt(width)

		qq = q.unsqueeze(1).expand([-1,n_ctx,-1,-1,-1])
		kk = k.unsqueeze(2).expand([-1,-1,n_ctx,-1,-1])

		ww = torch.abs(qq - kk)*scale
		attn = torch.sum(ww, -1) # sum over width

		ctx.save_for_backward(q, k)

		return attn

	@staticmethod
	def backward(ctx, d_attn):
		q, k = ctx.saved_tensors[:2]
		bs, n_ctx, n_heads, width = q.shape # shape!
		scale = -1.0 / math.sqrt(width)

		# recreate the expanded variables
		qq = q.unsqueeze(1).expand([-1,n_ctx,-1,-1,-1])
		kk = k.unsqueeze(2).expand([-1,-1,n_ctx,-1,-1])

		ws = torch.sign(qq - kk)*scale

		# pdb.set_trace()
		d_q = torch.einsum("bsthw,bsth->bthw", ws, d_attn) # sum over s index
		d_k = torch.einsum("bsthw,bsth->bshw", ws, -1*d_attn) # sum over t index

		return d_q, d_k


# class L1AttnF(nn.Module):
# 	def __init__(self):
# 		super(L1AttnF, self).__init__()
# 
# 	def forward(self, q, k):
# 		return L1AttnFunction.apply(q, k)
