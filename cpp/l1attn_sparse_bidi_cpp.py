import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import pdb

import l1attn_sparse_bidi_drv

class L1AttnSparseBidiFn(Function):
	@staticmethod
	def forward(ctx, vf, vb, q, k, coo, dst_mxlen, use_softmax):
		# bs, n_ctx, n_heads, width = q.shape
		vf = vf.contiguous(); 
		vb = vb.contiguous();
		q = q.contiguous(); 
		k = k.contiguous();
		vo,attn = l1attn_sparse_bidi_drv.forward(vf, vb, q, k, coo, dst_mxlen, use_softmax)
		ctx.save_for_backward(vf, vb, q, k, coo, attn, torch.tensor(dst_mxlen), torch.tensor(use_softmax))

		if False: # debug!
			# check that the attention is correct
			bs, n_tok, n_heads, width = q.shape
			cl = coo.shape[0] 
			
			qq = q[:,coo[:,0],:,:] # broadcast dst to cl
			kk = k[:,coo[:,1],:,:] # broadcast src to cl
			scale = -1 / math.sqrt(width) # -1 for subsequent softmax
			ww = torch.sum(torch.abs(qq - kk), -1)*scale
			attnp = torch.ones((bs, n_tok, dst_mxlen, n_heads),\
				device=q.device, dtype=q.dtype)*-1e12 # -infty
			attnp[:,coo[:,0],coo[:,2],:] = ww[:,0:cl,:] # scatter op
			attnp_sm = F.softmax(attnp, 2)
			pdb.set_trace()
			assert(torch.allclose(attnp_sm, attn))
			print('python:',attnp_sm)
			print('cpp:',attn)

		return vo

	@staticmethod
	def backward(ctx, dvo):
		# dvo is always contiguous?  guess so.
		vf,vb,q,k,coo,attn,dst_mxlen,use_softmax = ctx.saved_tensors[:8]
		dst_mxlen = dst_mxlen.item()
		use_softmax = use_softmax.item()
		
		d_vf, d_vb, d_q, d_k = l1attn_sparse_bidi_drv.backward(dvo, vf,vb,q,k,coo,attn,dst_mxlen,use_softmax)
		return d_vf, d_vb, d_q, d_k, None, None, None


class L1AttnSparseBidi(nn.Module):
	def __init__(self):
		super(L1AttnSparseBidi, self).__init__()

	def forward(self, q, k):
		return L1AttnSparseBidiFn.apply(q, k)
