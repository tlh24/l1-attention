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

		qq = q.permute(0, 2, 3, 1).unsqueeze(-2).expand([-1,-1,-1,n_ctx,-1])
		kk = k.permute(0, 2, 3, 1).unsqueeze(-1).expand([-1,-1,-1,-1,n_ctx])

		ww = torch.abs(qq - kk)*scale
		attn = torch.sum(ww, 2) # sum over width
		# NB: must do softmax over second-to-last dim:
		'''
		m = l1attn.L1Attn()
		a = m.forward(q, k)
		a_sm = F.softmax(a, -2)
		vo = torch.einsum('bhsd, bshw -> bhdw', a_sm, v)
		'''

		return attn
