import math
import torch
import torch.nn.functional as F


class L1Attn(torch.nn.Module):
	def __init__(self):
		super(L1Attn, self).__init__()
		# there are no parameters, deterministic mapping

	def forward(self, q, k):
		bs, n_ctx, n_heads, width = q.shape
		scale = -1 / math.sqrt(width)

		qq = q.unsqueeze(1).expand([-1,n_ctx,-1,-1,-1]) 
		kk = k.unsqueeze(2).expand([-1,-1,n_ctx,-1,-1])
		# really should switch this to match sparse attention. 
		# maybe when i have more free time... 

		ww = torch.abs(qq - kk)*scale
		attn = torch.sum(ww, -1) # sum over width
		# attn dimensions bs, src, dst, heads
		# NB: must do softmax over second dim:
		'''
		m = l1attn.L1Attn()
		a = m.forward(q, k)
		a_sm = F.softmax(a, 1)
		vf = torch.einsum('bsdh, bshw -> bdhw', a_sm, v)
		# where d = dst, query and s = src, key
		'''

		return attn
