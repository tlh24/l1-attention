import torch
from torch.autograd import gradcheck
from python.l1attn import L1Attn
import python.l1attn_baseline
import l1attn_cpp # must be installed!
import l1attn_cuda
import pdb
import time

def absum(a): 
    return torch.sum(torch.abs(a))

device = torch.device("cpu")
torch.manual_seed(int(time.time()))

kwargs = {'dtype': torch.float64, 
          'device': device,
          'requires_grad': True}
          
debug = False

if debug:
	batch_size = 1
	n_ctx = 3
	n_heads = 2
	width = 2
else: 
	batch_size = 2
	n_ctx = 3
	n_heads = 5
	width = 7

# # CUDA implemetation of IEEE-754 is not *exactly* like Intel / AMD's
# # but, if we restrict to multiples of 1/128, everything matches bit-perfectly. 
# # finer granulations results in errors on the order of 1e-16. 
# # ??perhaps there is a bug in the code?? 
# denom = 128
# q = torch.randint(0, denom, (batch_size, n_ctx, n_heads, width), **kwargs)/denom
# k = torch.randint(0, denom, (batch_size, n_ctx, n_heads, width), **kwargs)/denom
q = torch.randn((batch_size, n_ctx, n_heads, width), **kwargs)
k = torch.randn((batch_size, n_ctx, n_heads, width), **kwargs)

variables = [q, k]

l1a = L1Attn()
attn_naive = l1a(q,k)

attn_baseline = python.l1attn_baseline.L1AttnFn.apply(*variables)

attn_cpp = l1attn_cpp.L1AttnFn.apply(*variables)

variables_cuda = [q.cuda(), k.cuda()]
attn_cuda = l1attn_cuda.L1AttnFn.apply(*variables_cuda)

assert(torch.allclose(attn_naive, attn_baseline))
print('Forward: Baseline vs Naive Ok')

assert(torch.allclose(attn_naive, attn_cpp))
print('Forward: Cpp vs Naive Ok')

assert(torch.allclose(attn_naive, attn_cuda.cpu()))
print('Forward: Cuda vs Naive Ok')

# # note: the gradient dosen't work perfectly with rational numbers (above)
# # because the gradient isn't defined for abs(0)
# # (which occurs with rational numbers)
# q = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
# k = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
# variables = [q, k]

if gradcheck(python.l1attn_baseline.L1AttnFn.apply, variables):
    print('Backward: Baseline grad Ok')
    
if gradcheck(l1attn_cpp.L1AttnFn.apply, variables):
   print('Backward: Cpp grad Ok')

if gradcheck(l1attn_cuda.L1AttnFn.apply, variables_cuda):
   print('Backward: Cuda grad Ok')
