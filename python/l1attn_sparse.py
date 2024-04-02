import math
import torch
import torch.nn.functional as F
import pdb
import l1attn


class L1AttnSparse(torch.nn.Module):
	def __init__(self):
		super(L1AttnSparse, self).__init__()
		# there are no parameters, deterministic mapping

	def forward(self, q, k, v, coo, coo_cnt_max):
		'''
		q, k, v are the usual dense tensors 
			shape [batch_size, n_tok, n_heads, width]
			for Query, Key, and Value respectively. 
		coo is a vector size [cl,3]
			with elements coo[i,:] = [dst,src,sm_cnt]
			where dst indexes q
			and src indexes k,v
			and sm_cnt indexes softmax
				that is, for each dst it compresses/indexes src to be non-sparse.
				(otherwise we need to allocate a full softmax matrix)
		dst and src are in [0 .. n_tok)
		sm_cnt is in [0 .. coo_cnt_max)
		'''
		bs, n_tok, n_heads, width = q.shape
		cl = coo.shape[0] # tempted to name it cool (coo_length)
		qq = q.permute(0, 2, 1, 3) # batch, heads, n_ctx, width
		kk = k.permute(0, 2, 1, 3) 
		vv = v.permute(0, 2, 1, 3) 

		qq = qq[:,:,coo[:,0],:] # this should broadcast properly.
		kk = kk[:,:,coo[:,1],:]
		scale = -1 / math.sqrt(width) # -1 for subsequent softmax
		ww = torch.sum(torch.abs(qq - kk)*scale, -1)
		attn = torch.ones(bs, n_heads, n_tok, coo_cnt_max+1, device=q.device)*-1e32 # -infty
		attn[:,:,coo[:,0], coo[:,2]] = ww[:,:,0:cl] # scatter op
		attn_sm = F.softmax(attn, -1)
		vw = torch.zeros(bs, n_heads, n_tok, coo_cnt_max+1, width, device=q.device)
		vw[:,:,coo[:,0],coo[:,2],:] = vv[:,:,coo[:,1],:]
		vo = torch.einsum("bhds, bhdsw -> bdhw", attn_sm, vw) # sum over src
		vout = torch.zeros_like(v)
		vout[:,coo[:,0],:,:] = vo[:,coo[:,0],:,:]
		return vout

def expandCoo(co):
	'''
	take a coordinate vector 'co'
	consisting of [dst,src] pairs
	and add a third dimension for the softmax
	over source, per dest.
	'''
	coo = torch.zeros((co.shape[0], 3), dtype=torch.int32, device=co.device)
	cntr = {}
	coo_cnt_max = 0
	dst_max = 0
	for i in range(co.shape[0]):
		dst = co[i,0].item()
		src = co[i,1].item()
		if dst in cntr:
			cntr[dst] = cntr[dst] + 1
		else:
			cntr[dst] = 0
		coo[i,0] = dst
		coo[i,1] = src
		coo[i,2] = cntr[dst]
		coo_cnt_max = max(coo_cnt_max, cntr[dst])
		dst_max = max(dst_max, dst)
	# go back and make sure all destinations are written -
	# that is, all destinations have at least one source.
	for i in range(dst_max):
		if i not in cntr:
			print(f'degenerate sparse head - {i} not written')
	return coo, coo_cnt_max

def testL1AttnSparse(q, k, v, co):
	coo, coo_cnt_max = expandCoo(co)
	m = L1AttnSparse()
	vout = m(q, k, v, coo, coo_cnt_max)
	print('q', torch.squeeze(q))
	print('k', torch.squeeze(k))
	print('v', torch.squeeze(v))
	print('coo', coo)
	print('vout', torch.squeeze(vout))
	return vout

if __name__ == "__main__":
	batch_size = 1
	n_ctx = 3
	n_heads = 1
	width = 3
	
	q = torch.zeros(batch_size, n_ctx, n_heads, width)
	q[:,0,:,0] = 0
	q[:,0,:,1] = 1
	q[:,0,:,2] = 2
	q[:,1,:,0] = 2
	q[:,1,:,1] = 1
	q[:,1,:,2] = 0
	q[:,2,:,0] = 0
	q[:,2,:,1] = 0
	q[:,2,:,2] = 0
	
	k = torch.zeros(batch_size, n_ctx, n_heads, width)
	k[:,0,:,0] = 2
	k[:,0,:,1] = 1
	k[:,0,:,2] = 0
	k[:,1,:,0] = 0
	k[:,1,:,1] = 1
	k[:,1,:,2] = 2
	k[:,2,:,0] = 0
	k[:,2,:,1] = 1
	k[:,2,:,2] = 2
	
	v = torch.zeros(batch_size, n_ctx, n_heads, width)
	v[:,0,:,0] = -2
	v[:,0,:,1] = 2
	v[:,0,:,2] = 3
	v[:,1,:,0] = 2
	v[:,1,:,1] = -2
	v[:,1,:,2] = 2
	v[:,2,:,0] = 3
	v[:,2,:,1] = 2
	v[:,2,:,2] = -2

	co = torch.tensor([[0,0],[0,1],[1,0],[1,1],[2,2]])

	testL1AttnSparse(q, k, v, co)

	# try full non-sparse attention
	co = torch.tensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
	vs = testL1AttnSparse(q, k, v, co)
	# compare it with non-sparse L1 attention.
	m = l1attn.L1Attn()
	a = m.forward(q, k)
	a_sm = F.softmax(a, -2)
	vf = torch.einsum('bhsd, bshw -> bdhw', a_sm, v)
	print('full / default attn')
	print('vout', vf)
	print('diff', vs-vf)
	assert torch.allclose(vs, vf)

	# same thing, but permute the coo vector
	indx = torch.randperm(co.shape[0])
	co = co[indx, :]
	vs = testL1AttnSparse(q, k, v, co)
	assert torch.allclose(vs, vf)

	print('assertions passed')
