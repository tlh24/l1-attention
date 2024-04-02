import math

from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import pdb


class L1AttnFunction(Function):
	@staticmethod
	def forward(ctx, q, k):
		bs, n_ctx, n_heads, width = q.shape
		scale = -1 / math.sqrt(width)

		qq = q.permute(0, 2, 3, 1).unsqueeze(-2).expand([-1,-1,-1,n_ctx,-1])
		kk = k.permute(0, 2, 3, 1).unsqueeze(-1).expand([-1,-1,-1,-1,n_ctx])

		cc = torch.abs(qq - kk)*scale
		c = torch.sum(cc, 2) # sum along the width variable

		ctx.save_for_backward(q, k, c)

		return c

	@staticmethod
	def backward(ctx, d_attn):
		q, k, c = ctx.saved_tensors[:3]
		bs, n_ctx, n_heads, width = q.shape
		scale = 1.0 / math.sqrt(width)

		# recreate the expanded variables
		qq = q.permute(0, 2, 3, 1).unsqueeze(-2).expand([-1,-1,-1,n_ctx,-1])
		kk = k.permute(0, 2, 3, 1).unsqueeze(-1).expand([-1,-1,-1,-1,n_ctx])
		# shape: bs, n_heads, width, n_ctx, n_ctx

		ws = torch.sign(qq - kk)*scale

		d_r = 1 * c * d_attn

		d_q = torch.einsum("bhwst,bhst->bthw", ws, d_r) # sum over s index
		d_k = torch.einsum("bhwst,bhst->bshw", ws, -1*d_r) # sum over t index

		return d_q, d_k


class L1AttnF(nn.Module):
	def __init__(self):
		super(L1AttnF, self).__init__()

	def forward(self, q, k):
		return L1AttnFunction.apply(q, k)
