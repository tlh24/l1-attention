import math
import torch
import torch.nn.functional as F
import pdb

class L1AttnSparse(torch.nn.Module):
	def __init__(self):
		super(L1AttnSparse, self).__init__()
		# there are no parameters, deterministic mapping

	def forward(self, q, k, v, indx):
		bs, n_ctx, n_heads, width = q.shape
		n_tok = indx.shape[0]
		scale = 1 / math.sqrt(n_tok)

		qq = q.permute(0, 2, 3, 1) # batch, heads, width, n_ctx
		qq = qq[:,:,:,indx]
		qq = qq.unsqueeze(-2).expand([-1,-1,-1,n_tok,-1]) 
		# these attention matrices can be rectangular
		# for asymmetric (directed) dependencies!
		
		kk = k.permute(0, 2, 3, 1)
		kk = kk[:,:,:,indx]
		kk = kk.unsqueeze(-1).expand([-1,-1,-1,-1,n_tok])

		ww = torch.abs(qq - kk)*scale
		attn = -1.0 * torch.sum(ww, 2) # sum over width
		# -1 bc/ we'll be doing a softmax afterward. 
		
		attn = F.softmax(attn, -1) # softmax over the keys
		vv = torch.einsum("bthw, bhst -> bshw", v[:,indx,:,:], attn)

		return vv

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
	vv = m(q, k, v, torch.tensor([0,1]))
	print('q', torch.squeeze(q))
	print('k', torch.squeeze(k))
	print('v', torch.squeeze(v))
	print('vout', torch.squeeze(vv))
