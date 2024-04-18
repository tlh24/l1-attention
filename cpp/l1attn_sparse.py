import math
from torch import nn
from torch.autograd import Function
import torch
import pdb

import l1attnSparse_cpp


class L1AttnSparseFunction(Function):
	@staticmethod
	def forward(ctx, v, q, k, coo, dst_mxlen):
		# bs, n_ctx, n_heads, width = q.shape
		q = q.contiguous(); 
		k = k.contiguous();
		vo,attn = l1attnSparse_cpp.forward(v, q, k, coo, dst_mxlen)
		ctx.save_for_backward(v, q, k, coo, attn, torch.tensor(dst_mxlen))

		return vo

	@staticmethod
	def backward(ctx, dvo):
		v,q,k,attn,coo,dst_mxlen = ctx.saved_tensors[:6]
		dst_mxlen = dst_mxlen.item()
		
		d_v, d_q, d_k = l1attnSparse_cpp.backward(dvo, v,q,k,coo,attn,dst_mxlen)
		return d_v, d_q, d_k


class L1Attn(nn.Module):
	def __init__(self):
		super(L1Attn, self).__init__()

	def forward(self, q, k):
		return L1AttnFunction.apply(q, k)
