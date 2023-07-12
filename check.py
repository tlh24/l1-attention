import torch
from torch.autograd import gradcheck
from python.l1attn import L1Attn
from python.l1attn_baseline import L1AttnFunction
import cpp.l1attn
import cuda.l1attn
import pdb
import time

def absum(a): 
    return torch.sum(torch.abs(a))

device = torch.device("cpu")
torch.manual_seed(int(time.time()))

kwargs = {'dtype': torch.float64, # TODO: parameterize this later! 
          'device': device,
          'requires_grad': True}
          
batch_size = 2
n_heads = 4
n_ctx = 16
width = 16

# CUDA implemetation of IEEE-754 is not *exactly* like Intel / AMD's
# but, if we restrict to multiples of 1/128, everything matches bit-perfectly. 
# finer granulations results in errors on the order of 1e-16. 
# ??perhaps there is a bug in the code?? 
denom = 128
q = torch.randint(0, denom, (batch_size, n_ctx, n_heads, width), **kwargs)/denom
k = torch.randint(0, denom, (batch_size, n_ctx, n_heads, width), **kwargs)/denom
variables = [q, k]

l1a = L1Attn()
attn_naive = l1a(q,k)

attn_baseline = L1AttnFunction.apply(*variables)

attn_cpp = cpp.l1attn.L1AttnFunction.apply(*variables)

variables_cuda = [q.cuda(), k.cuda()]
attn_cuda = cuda.l1attn.L1AttnFunction.apply(*variables_cuda)

def check_forward(s, a): 
    b = absum(a).item(); 
    ok = "Ok" if(b == 0.0) else "not OK, absum %f" % b
    print(s, ok)

check_forward("Forward attn, absum( Naive - Baseline )", attn_naive - attn_baseline )
check_forward("Forward attn, absum( Naive - Cpp )", attn_naive - attn_baseline )
check_forward("Forward attn, absum( Naive - Cuda )", attn_naive - attn_cuda.cpu() )


# note: the gradient dosen't work perfectly with rational numbers (above)
# because the gradient isn't defined for abs(0)
# (which occurs with rational numbers)
q = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
k = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
variables = [q, k]

if gradcheck(L1AttnFunction.apply, variables):
    print('Backward: Baseline grad Ok')
    
if gradcheck(cpp.l1attn.L1AttnFunction.apply, variables):
    print('Backward: Cpp grad Ok')

if gradcheck(cuda.l1attn.L1AttnFunction.apply, variables_cuda):
    print('Backward: Cuda grad Ok')
