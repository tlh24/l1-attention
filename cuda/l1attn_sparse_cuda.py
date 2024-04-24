import math
from torch import nn
from torch.autograd import Function
import torch
import l1attn_sparse_cuda

class L1AttnSparseFn(Function):
	@staticmethod
	def forward(ctx, v, q, k, coo, dst_mxlen):
		# bs, n_ctx, n_heads, width = q.shape
		v = v.contiguous();
		q = q.contiguous();
		k = k.contiguous();
		vo,attn = l1attn_sparse_cuda.forward(v, q, k, coo, dst_mxlen)
		ctx.save_for_backward(v, q, k, coo, attn, torch.tensor(dst_mxlen))

		# check that the attention is correct
		bs, n_tok, n_heads, width = q.shape
		cl = coo.shape[0] # tempted to name it cool (coo_length)

		if False: # debug!
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
			print('cuda:',attn)

		return vo

	@staticmethod
	def backward(ctx, dvo):
		# dvo is always contiguous?  guess so.
		v,q,k,coo,attn,dst_mxlen = ctx.saved_tensors[:6]
		dst_mxlen = dst_mxlen.item()

		d_v, d_q, d_k = l1attn_sparse_cuda.backward(dvo, v,q,k,coo,attn,dst_mxlen)
		return d_v, d_q, d_k, None, None


class L1AttnSparse(nn.Module):
    def __init__(self):
        super(L1AttnSparse, self).__init__()

    def forward(self, q, k):
        return L1AttnSparseFn.apply(q, k)
