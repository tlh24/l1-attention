import math
from torch import nn
from torch.autograd import Function
import torch
import l1attn_sparse_bidi_cuda_drv


class L1AttnSparseBidiFn(Function):
	@staticmethod
	def forward(ctx, vf, vb, q, k, coo, dst_mxlen, use_softmax):
		# bs, n_ctx, n_heads, width = q.shape
		vf = vf.contiguous();
		vb = vb.contiguous();
		q = q.contiguous();
		k = k.contiguous();
		vo,attn = l1attn_sparse_bidi_cuda_drv.forward(vf, vb, q, k, coo, dst_mxlen, use_softmax)
		ctx.save_for_backward(vf, vb, q, k, coo, attn, torch.tensor(dst_mxlen), torch.tensor(use_softmax))

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
		# dvo is always contiguous?  not always! 
		dvo = dvo.contiguous()
		vf,vb,q,k,coo,attn,dst_mxlen,use_softmax = ctx.saved_tensors[:8]
		dst_mxlen = dst_mxlen.item()
		use_softmax = use_softmax.item()

		d_vf, d_vb, d_q, d_k = l1attn_sparse_bidi_cuda_drv.backward(dvo, vf,vb,q,k,coo,attn,dst_mxlen,use_softmax)
		return d_vf, d_vb, d_q, d_k, None, None, None


class L1AttnSparseBidi(nn.Module):
    def __init__(self):
        super(L1AttnSparseBidi, self).__init__()

    def forward(self, vf, vb, q, k, coo, dst_mxlen, use_softmax):
        return L1AttnSparseBidiFn.apply(vf, vb, q, k, coo, dst_mxlen, use_softmax)
	  
	  
def expandCoo(co):
	'''
	take a coordinate vector 'co'
	consisting of [dst,src] pairs
	- add a third dimension for the softmax
		over source, per dest.
	- add a fourth dimension for the backward pass
		over dest, per source
	'''
	coo = torch.zeros((co.shape[0], 4), dtype=torch.int32, device=co.device)
	dst_cntr = {}
	src_cntr = {}
	dst_mxlen = 0
	src_mxlen = 0
	dst_max = 0
	src_max = 0
	for i in range(co.shape[0]):
		dst = co[i,0].item()
		src = co[i,1].item()
		if dst in dst_cntr:
			dst_cntr[dst] = dst_cntr[dst] + 1
		else:
			dst_cntr[dst] = 0
		if src in src_cntr:
			src_cntr[src] = src_cntr[src] + 1
		else:
			src_cntr[src] = 0
		coo[i,0] = dst
		coo[i,1] = src
		coo[i,2] = dst_cntr[dst]
		coo[i,3] = src_cntr[src]
		dst_mxlen = max(dst_mxlen, dst_cntr[dst])
		src_mxlen = max(src_mxlen, src_cntr[src])
		dst_max = max(dst_max, dst)
		src_max = max(src_max, src)
	# go back and make sure all destinations are written -
	# that is, all destinations have at least one source.
	for i in range(dst_max):
		if i not in dst_cntr:
			print(f'Warning: degenerate sparse head - {i} not written')
	for i in range(src_max):
		if i not in src_cntr:
			print(f'Warning degenerate sparse head - {i} not read')
	# print('coo', coo)
	# dst_mxlen and src_mxlen are indexes / add 1 to get the max length.
	return coo, dst_mxlen+1, src_mxlen+1
