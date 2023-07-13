from collections import OrderedDict
import math
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import l1attn

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
	  
class ResidualAttentionBlock(nn.Module):
	def __init__(self, d_model: int, n_head: int, use_l1attn: bool):
		super().__init__()

		self.n_heads = n_head
		self.use_l1attn = use_l1attn
		self.c_qkv = nn.Linear(d_model, d_model * 3)
		self.ln_1 = nn.LayerNorm(d_model)
		self.attn = l1attn.L1Attn()
		self.mlp = nn.Sequential(OrderedDict([
			("c_fc", nn.Linear(d_model, d_model * 4)),
			("gelu", QuickGELU()),
			("c_proj", nn.Linear(d_model * 4, d_model))
		]))
		self.ln_2 = nn.LayerNorm(d_model)

	def attention_dp(self, x : torch.Tensor): 
		# this is faster than pytorch built-in MultiheadAttention! 
		qkv = self.c_qkv(x)
		bs, n_ctx, width = x.shape
		attn_ch = width // self.n_heads 
		scale = 1 / math.sqrt(math.sqrt(attn_ch))
		qkv = qkv.view(bs, n_ctx, self.n_heads, -1) 
		# shape: bs, ctx, n_heads, attn_ch*3 
		q, k, v = torch.split(qkv, attn_ch, dim=-1)
		weight = torch.einsum(
			"bthc,bshc->bhts", q * scale, k * scale
		)  # More stable with f16 than dividing afterwards
		wdtype = weight.dtype
		k = torch.arange(0,n_ctx)
		weight[:,:,k,k] = -10.0; # zero the diagonal, to make it fair! 
		weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
		return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

	def attention_l1(self, x : torch.Tensor): 
		qkv = self.c_qkv(x)
		bs, n_ctx, width = x.shape
		attn_ch = width // self.n_heads 
		scale = 1 / math.sqrt(attn_ch) # does not seem to have much effect
		qkv = qkv.view(bs, n_ctx, self.n_heads, -1) 
		# shape: bs, ctx, n_heads, attn_ch*3 
		q, k, v = torch.split(qkv, attn_ch, dim=-1)
		weight = self.attn(q, k)
		k = torch.arange(0,n_ctx)
		weight[:,:,k,k] = 0.0; # zero the diagonal; could do this in cuda..
		wdtype = weight.dtype
		weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
		return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

	def forward(self, x: torch.Tensor):
		if(self.use_l1attn):
			x = x + self.attention_l1(x) #self.ln_1(x) 
		else:
			x = x + self.attention_dp(x) #self.ln_1(x) 
		x = x + self.mlp(x) #self.ln_2(x)
		return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, use_l1attn: bool):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, use_l1attn) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Recognizer(nn.Module): 
	def __init__(
		self,
		n_ctx:int,
		n_layers:int, 
		n_heads:int, 
		indim:int,
		embed_dim:int,
		use_l1attn:bool
		): 
		super().__init__()
		self.n_ctx = n_ctx
		self.indim = indim
		self.embed_dim = embed_dim
		
		self.encoder = nn.Linear(indim, embed_dim)
		self.gelu = QuickGELU()
		
		self.trfmr = Transformer(
			width = embed_dim, 
			layers = n_layers, 
			heads = n_heads, 
			use_l1attn = use_l1attn)
		
		self.decoder = nn.Linear(embed_dim, 1)
		
	def forward(self, x, use_l1attn): 
		x = self.encoder(x)
		x = self.gelu(x)
		x = self.trfmr(x) # [bs, n_ctx, embed_dim]
		x = self.decoder(x) # [bs, n_ctx, 1]
		return x[:,:,0].sum(1)
