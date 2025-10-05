# want to test the L1 transformer on a few basic tasks
# e.g. is an element in the set of tokens? 
# if so, what is the distance to match a given pattern? 
# can we learn this purely from the distance or abs distance? 
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import l1attn_cuda
import matplotlib.pyplot as plt
import pdb
# import psgd 
import psgd_20240805 as psgd
import time

class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)
	  
class LinearM(nn.Module): 
	def __init__(self, indim:int, outdim:int, initzeros:bool): 
		super(LinearM, self).__init__()
		scl = 0.005
		if initzeros: 
			self.w = torch.nn.Parameter( scl * torch.ones(outdim,indim+1))
		else:
			self.w = torch.nn.Parameter( scl * torch.randn(outdim, indim+1))
		with torch.no_grad(): 
			self.w[:,-1] = 0.0 # bias starts at 0
		
	def forward(self, x):
		return torch.einsum('oi,bhi -> bho', self.w[:,:-1], x) + self.w[:,-1]

def graycodePosEnc(ntok, nbits, rand_phase=False):
	'''
	Generate a graycode
	seems more principled than standard SPE?
	'''
	pos_enc = np.zeros((ntok,nbits*2), dtype=np.float32)
	indx = np.linspace(0, (ntok-1)*2*math.pi, ntok)
	if rand_phase:
		phase_offset = np.random.uniform() * 2 * math.pi
	else:
		phase_offset = 0
	for i in range(nbits):
		# gray code: [0][1] has a period of 4
		# [2][3] has a period of sqrt(4*8) = sqrt(32) = 4 sqrt(2)
		# [4][5] period of 8..
		# period = 4 * (math.sqrt(2.0))**i
		#   above is slower - does not help?
		period = 4 * 2.0**i
		if True:
			# sinusoidal, seems to work better?
			pos_enc[:, 2*i  ] = -np.cos(indx / period + phase_offset)
			pos_enc[:, 2*i+1] = np.sin(indx / period + phase_offset)
		else:
			# graycode! (thresholded)
			pos_enc[:, 2*i  ] = np.cos(indx / period + phase_offset) < 0
			pos_enc[:, 2*i+1] = np.sin(indx / period + phase_offset) < 0
	return pos_enc

def genData(bs, npos, width, cursor, cmd_args):
	ntok = npos + 1
	target = torch.zeros(bs,width)
	lin = torch.arange(0,npos)
	x = torch.zeros(bs,ntok, width) # position encoding
	# binary encoding of the digits.  
	x[:,:npos,0:16] = torch.tensor(graycodePosEnc(npos, 8)*10)
	x[:,:npos, 16] = 10 # indicate this is the set: search over these
	x[:,:npos, 17:-1] = torch.randn(bs,npos, width-18)*5
	x[:,npos, -1] = 10 # where we measure the loss.
	target = torch.zeros(bs,width)
	idx = torch.arange(bs)
	target[idx, 17:32] = x[idx, cursor, 17:32]
	return x,target
	
	
class ResidualAttentionBlock(nn.Module): 
	def __init__(self, d_model: int, n_head: int, init_zeros:bool):
		super().__init__()
		
		self.n_head = n_head
		self.d_model = d_model
		self.init_zeros = init_zeros
		
		self.wq = LinearM(d_model, n_head*d_model, init_zeros) 
		self.wv = LinearM(d_model, n_head*d_model, init_zeros)
		self.wk = torch.nn.Parameter( 0.005 * torch.ones(n_head, d_model) )
		
		self.l1a_f = l1attn_cuda.L1Attn() # dense or full attention
		self.soft = torch.nn.Softmax(dim=2) # unused with L1 attn
		self.fanout = LinearM(d_model, d_model * 1, False)
		# self.gelu = QuickGELU()
		self.gelu = nn.ReLU()
		# self.gelu = nn.LeakyReLU()
		
	def attention(self, x:torch.Tensor, axs):
		# pdb.set_trace()
		n_head = self.n_head
		d_head = self.d_model ## no sub-spaces!
		
		# x is [batch, tokens, d_model]
		batch_size = x.shape[0]
		ntok = x.shape[1]
		width = x.shape[2]
		
		q = self.wq(x)
		q = torch.reshape(q, (batch_size, ntok, self.n_head, d_head))
		
		v = self.wv(x)
		v = torch.reshape(v, (batch_size, ntok, self.n_head, d_head))
		
		# per-axis gate k by wk, uniformly across tokens; different per head.
		# this should be information-preserving.
		k = x.unsqueeze(2).expand([-1,-1,self.n_head,-1])
		gk = self.wk.unsqueeze(0).unsqueeze(0)
		k = k * gk 
		
		# extract all global / all-to-all tokens
		# really could do this with pure sparse attn.. will have to compare. 
		a2len = q.shape[1]
		# pad out to BLKSIZ tokens (for CUDA kernel).
		padn = ((a2len + 15) // 16) * 16 - a2len
		assert(padn > 0) # for noop
		qq = torch.cat((q, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
		kk = torch.cat((k, torch.zeros(batch_size, padn, n_head, width, device=v.device)), axis=1)
		a = self.l1a_f(qq, kk) # includes 1 / sqrt(head)
		a = a[:, :a2len+1, :a2len, :]
		a[:, a2len, :,:] = 0.0 # slight improvement.. 
		# add in e^0=1 as a 'noop' option
		# (hence max attention is 0.5, not 1)
		# output is b,src,dst,heads
		a = F.softmax(a, 1) # see l1attn.py -- sm over src 
		a = a[:, :a2len, :a2len, :] # remove noop
		b = torch.einsum('bsdh, bshw -> bdhw', a, v)
		b = torch.sum(b, dim=2) # sum along the heads
		b = torch.reshape(b, (batch_size, ntok, self.d_model))
		
		if axs is not None: 
			for h in range(self.n_head):
				im = axs[0,h+1].imshow(qq[0,:,h,:].detach().squeeze().cpu().numpy())
				plt.colorbar(im, ax = axs[0,h+1])
				axs[0,h+1].set_title(f"qq")
				
				im = axs[1,h+1].imshow(kk[0,:,h,:].detach().squeeze().cpu().numpy())
				plt.colorbar(im, ax = axs[1,h+1])
				axs[1,h+1].set_title(f"kk")
				
				im = axs[2,h+1].imshow(v[0,:,h,:].detach().squeeze().cpu().numpy())
				plt.colorbar(im, ax = axs[2,h+1])
				axs[2,h+1].set_title(f"v")
				
				im = axs[3,h+1].imshow(a[0,:,:,h].detach().squeeze().cpu().numpy())
				plt.colorbar(im, ax = axs[3,h+1])
				axs[3,h+1].set_title(f"attn post softmax")
				
				im = axs[4,h+1].imshow(b[0,:,:].detach().squeeze().cpu().numpy())
				plt.colorbar(im, ax = axs[4,h+1])
				axs[4,h+1].set_title(f"output b")
			
		return b

	def forward(self, x:torch.Tensor, axs=None):
		y = self.attention(x, axs)
		y = self.gelu(y)
		y = self.fanout(y) # allow sign inversions & mixing; no dim change
		return x + y
		
	def plot(self, x): 
		fig,axs = plt.subplots(5, self.n_head+1, figsize=(20,20))
		h = 0
		im = axs[0,h].imshow(self.wq.w.detach().cpu().numpy())
		plt.colorbar(im, ax = axs[0,h])
		axs[0,h].set_title(f"query_{h}")
		
		im = axs[1,h].imshow(self.wk.detach().cpu().numpy())
		plt.colorbar(im, ax = axs[1,h])
		axs[1,h].set_title(f"key_{h}")
	
		im = axs[2,h].imshow(self.wv.w.detach().cpu().numpy())
		plt.colorbar(im, ax = axs[2,h])
		axs[2,h].set_title(f"value_{h}")

		im = axs[3,h].imshow(x[0,:,:].detach().cpu().numpy())
		plt.colorbar(im, ax = axs[3,h])
		axs[3,h].set_title(f"x")
		
		y = self.forward(x, axs)

		im = axs[4,h].imshow(y[0,:,:].detach().cpu().numpy())
		plt.colorbar(im, ax = axs[4,h])
		axs[4,h].set_title(f"y")
		
		plt.show()
	
	
class Transformer(nn.Module): 
	def __init__(self, d_model:int, layers:int, repeat:int, n_head:int, init_zeros:bool, gendata_dim:int):
		super().__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.layers = layers
		self.repeat = repeat
		self.resblocks = nn.ModuleList([ResidualAttentionBlock(d_model, n_head, init_zeros) for _ in range(layers)])
		self.in_proj = LinearM(gendata_dim, d_model, False)
		self.out_proj = LinearM(d_model, gendata_dim, False)

	def forward(self, x:torch.Tensor):
		x = self.in_proj(x)
		for i in range(self.repeat): 
			for j, layer in enumerate(self.resblocks):
				x = layer(x)
		return self.out_proj(x)

	def plot(self, x): 
		for j, layer in enumerate(self.resblocks):
			layer.plot(x)

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
	parser.add_argument('-b', type=int, default=128, help='training data size')
	parser.add_argument('-d', type=int, default=0, help='CUDA device')
	parser.add_argument('--layers', type=int, default=1, help='number of layers')
	parser.add_argument('--heads', type=int, default=2, help='number of heads')
	parser.add_argument('-a', action='store_true', help='use AdamW')
	parser.add_argument('-l', action='store_true', default=False, help='use PSGD LRA')
	parser.add_argument('-m', action='store_true', help='many-mode - limit reporting and write results to file. ')
	parser.add_argument('--npos', type=int, default=16, help='number of positions to search over')
	parser.add_argument('--distract', type=int, default=8, help='distractor dims')
	parser.add_argument('-t', action='store_true', help='do a test plot.')
	cmd_args = parser.parse_args()
	sample_size = cmd_args.b
	npos = cmd_args.npos
	ntok = npos + 4
	width = cmd_args.distract + 32

	cursor = np.random.randint(npos)
	
	if cmd_args.t:
		fig,axs = plt.subplots(3, 3, figsize=(20,20))
		for i in range(3): 
			for j in range(3):
				y,target = genData(1, npos, width, cursor, cmd_args)
				y = y.squeeze().numpy()
				target = target.squeeze().numpy()
				im = axs[i,j].imshow(y)
				plt.colorbar(im, ax = axs[i,j])
				axs[i,j].set_title(f"target:{target[0]}")
		plt.show()
		exit()
	
	start_time = time.time()
	duration = 300 
	j = 0
	smooth_loss = 0.0
	if not cmd_args.m: 
		j = 199
	while j < 200: 
		
		model = Transformer(d_model=160, layers=cmd_args.layers, repeat=1, n_head=cmd_args.heads, init_zeros=False, gendata_dim=width)
		model.printParamCount()
		# pdb.set_trace()
		# model.fixedInit()
		model = model.cuda(cmd_args.d)

		use_adam = cmd_args.a
		
		if use_adam:
			optimizer = optim.AdamW(model.parameters(), lr=2e-3, amsgrad=True)
		else:
			if cmd_args.l: 
				optimizer = psgd.LRA(model.parameters(),lr_params=0.01,lr_preconditioner= 0.01, momentum=0.9,\
					preconditioner_update_probability=0.5, exact_hessian_vector_product=False, rank_of_approximation=100, grad_clip_max_norm=5.0)
			else: 
				optimizer = psgd.Affine(model.parameters(),lr_params=0.01,lr_preconditioner=0.01, momentum=0.9,\
					preconditioner_max_size=200000, \
					preconditioner_max_skew=200000, \
					preconditioner_update_probability=0.5, exact_hessian_vector_product=False, grad_clip_max_norm=5.0)
			# turning on exact_hessian_vector_product makes things worse! 
			# increasing the learning rate to 0.03 while keeping lr_preconditioner at 0.01 works well! usually converges.
			# seems we can tolerate a large lr since we're operating on all the data - no mini-batches! 
			# whitening preconditioner does not work. 
			# optimizer = psgd.XMat(model.parameters(),lr_params=0.01,lr_preconditioner= 0.01, momentum=0.9,\
			# 	preconditioner_update_probability=0.1, exact_hessian_vector_product=False, grad_clip_max_norm=5.0, preconditioner_type="Newton")
		
		fd_losslog = open('losslog.txt', 'w')
		
		# dat,_ = genData(2000, npos, cmd_args)
		# mean = torch.sum(dat, (0,1)) / (2000 * ntok)
		# std = torch.sqrt(torch.sum((dat - mean)**2, (0,1)) / (2000 * ntok))
		# std = std / 3.5 # help l1 attn select one
		
		x,target = genData(sample_size, npos, width, cursor, cmd_args)
		x = x.cuda(cmd_args.d)
		target = target.cuda(cmd_args.d)
		
		for i in range(10000):
			xx = x
			targetx = target
			if use_adam:
				y = model(xx)
				loss = torch.sum( (y[:,-1,:] - targetx)**2 )
				torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
				loss.backward()
				optimizer.step()
			else: 
				def closure(): 
					y = model(xx)
					loss = torch.sum( (y[:,-1,:] - targetx)**2 ) # + \
							# sum( \
							# [torch.sum(5e-4 * torch.rand_like(param) * torch.abs(param) ) for param in model.parameters()])
					return loss
				loss = optimizer.step(closure) 
			lloss = loss.detach().cpu().item()
			smooth_loss = 0.95*smooth_loss + 0.05*lloss
		
			if not cmd_args.m: 
				if i % 10 == 0: 
					print(lloss)
					fd_losslog.write(f'{i}\t{lloss}\n')
					fd_losslog.flush()
				elapsed_time = time.time() - start_time
				if elapsed_time > duration :
					j = 200
					break
		if cmd_args.m:
			j = j + 67
		else:
			j = j + 1
			
	
		x,target = genData(1000, npos, width, cursor, cmd_args)
		# x = (x - mean) / std # learn the affine transform later
		x = x.cuda(cmd_args.d)
		target = target.cuda(cmd_args.d)
		y = model(x)
		loss = torch.sum( (y[:,-1,:] - target)**2 )
		lloss = loss.detach().cpu().item()
		print("v",lloss)
		if not cmd_args.m: 
			fd_losslog.write(f'{i}\t{lloss}\n')
			fd_losslog.flush()
	
		if cmd_args.m: 
			optstr = 'psgd'
			if cmd_args.a:
				optstr = 'adam'
			fd_vallog = open(f'vallog_x5_{optstr}_l{cmd_args.layers}_h{cmd_args.heads}_npos{npos}_width{width}.txt', 'a')
			fd_vallog.write(f'{sample_size}\t{lloss/1000}\t{smooth_loss/sample_size}\n')
			fd_vallog.flush()
			fd_vallog.close()
	
	# if not cmd_args.m: 
	# 	x,target = genData(sample_size, cmd_args)
	# 	x = x.cuda(cmd_args.d)
	# 	y = model.plot(x)
