import math

from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import pdb


class L1AttnFunction(Function):
	@staticmethod
	def forward(ctx, q, k):
		bs, n_ctx, n_heads, width = q.shape # shape!
		scale = -1 / math.sqrt(width)

		qq = q.permute(0, 2, 1, 3).unsqueeze(-3).expand([-1,-1,n_ctx,-1,-1])
		kk = k.permute(0, 2, 1, 3).unsqueeze(-2).expand([-1,-1,-1,n_ctx,-1])

		cc = torch.abs(qq - kk)*scale
		c = torch.sum(cc, -1) # sum along the width axis

		ctx.save_for_backward(q, k, c)

		return c

	@staticmethod
	def backward(ctx, d_attn):
		q, k, c = ctx.saved_tensors[:3]
		bs, n_ctx, n_heads, width = q.shape # shape!
		scale = 1.0 / math.sqrt(width)

		# recreate the expanded variables
		qq = q.permute(0, 2, 1, 3).unsqueeze(-3).expand([-1,-1,n_ctx,-1,-1])
		kk = k.permute(0, 2, 1, 3).unsqueeze(-2).expand([-1,-1,-1,n_ctx,-1])
		# shape: bs, n_heads, n_ctx, n_ctx, width

		ws = torch.sign(qq - kk)*scale

		d_r = d_attn

		# pdb.set_trace()
		d_q = torch.einsum("bhstw,bhst->bthw", ws, -1*d_r) # sum over s index
		d_k = torch.einsum("bhstw,bhst->bshw", ws, d_r) # sum over t index

		return d_q, d_k


class L1AttnF(nn.Module):
	def __init__(self):
		super(L1AttnF, self).__init__()

	def forward(self, q, k):
		return L1AttnFunction.apply(q, k)
