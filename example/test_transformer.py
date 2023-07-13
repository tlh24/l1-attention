import math
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import l1attn
from tqdm import tqdm
from model import Recognizer
import pdb

batch_size = 256
n_ctx = 16 # number of tokens. determines the problem size. 
n_symb = 10 # dimension of the tokens
n_layers = 2
n_heads = 4
n_test = 5  # how many replicate training runs
n_iters = 10000
learning_rate = 5e-4
clip_grad = 0.1

doplot = False
ringers = False

torch_device = 0
print("torch cuda devices", torch.cuda.device_count())
print("torch device", torch.cuda.get_device_name(torch_device))
torch.cuda.set_device(torch_device)
torch.set_default_device(torch_device)
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('high') # desktop.


def gen_data(): 
	# task: select the two symbols that match
	# subtract the difference between their position encodings
	# background symbol encodings -- random [0, 0.5)
	x = torch.zeros(batch_size, n_ctx, n_symb+5)
	x[:,:,0:n_symb] = torch.randn(batch_size, n_ctx, n_symb) / 2
	# select matches
	i = torch.randint(n_ctx, [batch_size])
	j = torch.randint(n_ctx-1, [batch_size]) + 1
	j = torch.remainder(i + j, n_ctx)
	k = torch.arange(batch_size)
	# set the i = j symbol match
	x[k,i,:] = x[k,j,:]
	# set 'special' symbols: sinusoids with random phase.
	if(ringers):
		ss = torch.sin(torch.arange(0, n_symb).unsqueeze(0).expand(batch_size,n_symb) + torch.rand(batch_size).unsqueeze(-1).expand(batch_size,n_symb)*6.28)*1
		cc = torch.cos(torch.arange(0, n_symb).unsqueeze(0).expand(batch_size,n_symb) + torch.rand(batch_size).unsqueeze(-1).expand(batch_size,n_symb)*6.28)*1
		s2 = ss.clone()
		s2[:,:n_symb] = s2[:,:n_symb] 
		x[k,i,0:n_symb] = ss[k,:]
		x[k,j,0:n_symb] = s2[k,:]
	# pdb.set_trace()
	# positions
	x[:,:,-5] = torch.randint(128, [batch_size, n_ctx]) / 16.0
	x[:,:,-4] = x[:,:,-5] / 4
	x[:,:,-3] = x[:,:,-4] / 4
	x[:,:,-2] = x[:,:,-3] / 4
	# labels for us humans ;)
	x[:,:,-1] = 0
	x[k,i,-1] = 1
	x[k,j,-1] = 1
	y = torch.abs(x[k,i,-5] - x[k,j,-5])
	return x,y
	
def test_plot (): 
	x,y = gen_data()
	print(y)
	plt.imshow(x[0,:,:].cpu().numpy())
	plt.clim(0,2)
	plt.colorbar()
	plt.show()
	
# test_plot() # check that the tensor 'makes sense'


def train_model(use_l1attn): 
	slowloss = 1.0

	kwargs = {'dtype': torch.float32, 
				'device': torch.device("cpu"),
				'requires_grad': False}
	testloss = torch.zeros(n_test, **kwargs)
	plotloss = torch.zeros(n_test, n_iters, **kwargs)

	for j in range(n_test): 
		# reinit the model each time. 
		model = Recognizer(n_ctx = n_ctx, 
							n_layers = n_layers, 
							n_heads = n_heads, 
							indim = n_symb+4,
							embed_dim = 64, 
							use_l1attn = use_l1attn)

		lossfunc_mse = nn.MSELoss(reduction='mean')
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)

		for i in tqdm(range(n_iters)): 
			x,targ = gen_data()
			
			if(doplot):
				plt.imshow(x[0,:,:].cpu().numpy())
				plt.clim(0,2)
				plt.colorbar()
				plt.show()
			
			xp = x[:,:,:-1] # remove the human annotations
			y = model(xp, use_l1attn)
			loss = lossfunc_mse(y,targ)
			lossflat = torch.sum(loss)
			lossflat.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
			optimizer.step() 
			lossflat.detach()
			slowloss = 0.99*slowloss + 0.01 * lossflat.item()
			plotloss[j,i] = slowloss
			# print(slowloss)
			
			# need to check that we can visually solve the problem! 
			
			if(doplot):
				# dotproduct between symbols
				w = torch.einsum("btc,bsc->bts", x[:,:,0:n_symb], x[:,:,0:n_symb])
				plt.imshow(w[0,:,:].cpu().numpy())
				plt.colorbar()
				plt.show()
			# get the best matches
			# pdb.set_trace()

		testloss[j] = slowloss
		
	return plotloss
	
l1_loss = train_model(True)
dp_loss = train_model(False)
	

l1_loss = l1_loss[:, 100:] # ignore the warm-up
dp_loss = dp_loss[:, 100:] # ignore the warm-up

fig, axs = plt.subplots(1, 2)

axs[0].plot(l1_loss.transpose(1,0).numpy())
axs[0].set_title('Loss w L1 attention')
axs[0].set(xlabel='iteration')
axs[1].plot(dp_loss.transpose(1,0).numpy())
axs[1].set_title('Loss w DP attention')
axs[1].set(xlabel='iteration')
plt.show()

