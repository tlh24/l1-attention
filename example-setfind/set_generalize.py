'''
really hard sudoku:
..9...2...8.5...1.7.......6..6.9.....5.8..3..4....7........4..9.3..1..8....2..5..

Want to see if the current L1 transformer permits generalization over axes:
If you learn indexing then intersection for one axis,
does it generalize to different axes?
Do this with a recurrent tansformer, so there is an opportunity to re-use the heads.
'''
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import l1attn_cuda
import matplotlib.pyplot as plt
import psgd
import pdb

width = 16

def genData(bs):
	x = np.zeros((bs, 4, 4, width)) # middle two dimensions will be flattened to 16
	y = np.zeros((bs, 4, width)) #
	for b in range(bs):
		for i in range(4):
			d = np.random.permutation(5)
			for j in range(4):
				e = d[j]
				x[b,i,j,e] = 1 # one-hot digit encoding
				x[b,i,j,5] = i
				x[b,i,j,6] = j
			e = d[4]
			y[b,i,e] = 1
			y[b,i,5] = i
	return x, y

class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
	def __init__(self, d_model: int, n_head: int):
		super().__init__()

		self.n_head = n_head
		self.d_model = d_model
		self.wk = nn.Parameter( 0.005 * torch.ones(n_head, d_model) )

		self.wqv = nn.Linear(d_model, 3*n_head*d_model)
		self.initWeights(self.wqv)
		self.fanin = nn.Linear(d_model, d_model)
		self.initWeights(self.wqv)

		self.l1a_f = l1attn_cuda.L1Attn()

		self.gelu = QuickGELU()

	def initWeights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.005) # FIXME
			if module.bias is not None:
					torch.nn.init.zeros_(module.bias)

	def attention(self, x:torch.Tensor):
		n_head = self.n_head
		d_head = self.d_model ## no sub-spaces!
		batch_size = x.shape[0]
		ntok = x.shape[1]
		width = x.shape[2]

		v = self.wqv(x)
		v = torch.reshape(v, (batch_size, ntok, 3*self.n_head, d_head))
		q,vf,vb = torch.split(v, self.n_head, 2)

		# per-axis gate k by wk, uniformly across tokens; different per head.
		# this should be information-preserving.
		k = x.unsqueeze(2).expand([-1,-1,self.n_head,-1])
		wk = self.wk.unsqueeze(0).unsqueeze(0)
		k = k * wk

		# normal dense attention over all tokens
		# pad out to BLKSIZ tokens (for CUDA kernel).
		padn = ((ntok + 15) // 16) * 16 - ntok
		assert(padn > 0) # for noop
		qq = torch.cat((q, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
		kk = torch.cat((k, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
		a = self.l1a_f(qq, kk) # includes 1 / sqrt(head)
		a = a[:, :ntok+1, :ntok, :]
		a[:, ntok, :,:] = 0.0 # slight improvement:
		# adds in e^0=1 as a 'noop' option
		# (hence max attention is 0.5, not 1)
		# a is [b,src,dst,heads]
		a = F.softmax(a, 1) # see l1attn.py -- sm over src
		a = a[:, :ntok, :ntok, :] # remove noop
		bf = torch.einsum('bsdh, bshw -> bdhw', a, vf)
		bb = torch.einsum('bdsh, bshw -> bdhw', a, vb) # note transpose!
		b = bf + bb
		b = torch.sum(b, dim=2) # sum along the heads
		b = torch.reshape(b, (batch_size, ntok, self.d_model))
		return b # residual sum later.

	def forward(self, x:torch.Tensor):
		y = self.attention(x)
		y = self.gelu(y)
		y = self.fanin(y) # allow sign inversions & mixing; no dim change
		return x + y

class Transformer(nn.Module):
	def __init__(self, d_model:int, layers:int, repeat:int, n_head:int):
		super().__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.layers = layers
		self.repeat = repeat
		self.resblocks = nn.ModuleList(\
			[ResidualAttentionBlock(d_model, n_head) \
				for _ in range(layers)])
		self.in_proj = nn.Linear(d_model, d_model, bias=True)
		self.out_proj = nn.Linear(d_model, d_model, bias=True)

	# @torch.compile
	def forward(self, x:torch.Tensor, hcoo:list):
		x = self.in_proj(x)
		for i in range(self.repeat):
			for j, layer in enumerate(self.resblocks):
				x = layer(x)
		return self.out_proj(x)

	def fixedInit(self):
		for layer in self.resblocks:
			layer.fixedInit()

	def printParamCount(self):
		trainable_params = sum(
			p.numel() for p in self.parameters() if p.requires_grad
		)
		print(f"Number of model parameters:{trainable_params}")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda', type=int, default=0, help='CUDA device')
	cmd_args = parser.parse_args()
	
	bs = 1
	x,y = genData(bs)
	x = np.squeeze(x)
	y = np.squeeze(y)
	x = x.reshape(16, width)
	y = y.reshape(4, width)
	print('x')
	print(x[:,:8])
	print('y')
	print(y[:,:8])

	model = Transformer(d_model=width, layers=1, repeat=1, n_head=1)
	model.printParamCount()
	model = model.cuda(cmd_args.cuda)

	optimizer = psgd.LRA(model.parameters(),lr_params=0.01,\
					lr_preconditioner= 0.01, momentum=0.9,\
					preconditioner_update_probability=0.5, \
					exact_hessian_vector_product=False, \
					rank_of_approximation=100, grad_clip_max_norm=5.0)
	
	fd_losslog = open('losslog.txt', 'w')
	
	datx,daty = genData(2000)


