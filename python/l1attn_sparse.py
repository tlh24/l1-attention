import math
import torch
import torch.nn.functional as F
import pdb



class L1AttnSparse(torch.nn.Module):
	def __init__(self):
		super(L1AttnSparse, self).__init__()
		# there are no parameters, deterministic mapping

	def forward(self, q, k, v, coo, sm_cnt_max):
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
		sm_cnt is in [0 .. sm_cnt_max)
		'''
		bs, n_tok, n_heads, width = q.shape
		cl = coo.shape[0] # tempted to name it cool (coo_length)
		qq = q.permute(0, 2, 1, 3) # batch, heads, n_ctx, width
		kk = k.permute(0, 2, 1, 3) 
		vv = v.permute(0, 2, 1, 3) 
		
		qq = qq[:,:,coo[:,0],:] # this should broadcast properly? 
		kk = kk[:,:,coo[:,1],:]
		scale = 1 / math.sqrt(sm_cnt_max) # ???
		ww = torch.sum(torch.abs(qq - kk)*scale, -1)
		attn = torch.zeros(bs, n_heads, n_tok, sm_cnt_max, device=q.device)
		attn[:,:,coo[:,0], coo[:,2]] = ww[:,:,0:cl] # scatter op
		attn_sm = F.softmax(attn, -1)
		vv[:,:,coo[:,0],coo[:,2],:] = vv[:,:,0:cl,:]
		vo = torch.einsum("bhds, bhdsw -> bdhw", attn_sm, vv)
		vout = torch.zeros_like(v)
		vout[:,coo[:,0],:,:] = vout[:,coo[:,0],:,:]
		
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
	v[:,0,:,0] = -0.5
	v[:,0,:,1] = 0.5
	v[:,0,:,2] = 1
	v[:,1,:,0] = 0.5
	v[:,1,:,1] = -0.5
	v[:,1,:,2] = 0.5
	v[:,2,:,0] = 1
	v[:,2,:,1] = 0.5
	v[:,2,:,2] = -0.5
	
	m = L1AttnSparse()
	vv = m(q, k, v, torch.tensor([0,1]), torch.tensor([0,1]))
	print('q', torch.squeeze(q))
	print('k', torch.squeeze(k))
	print('v', torch.squeeze(v))
	print('vout', torch.squeeze(vv))
