import torch
from torch.autograd import gradcheck
from python.l1attn import L1Attn
from python.l1attn_baseline import L1AttnFunction, L1AttnF
import cpp.l1attn
import cuda.l1attn
import pdb
import time

### this script is to help debugging CUDA gradient calculation
### (by priting out the gradient tensors.)

def absum(a): 
    return torch.sum(torch.abs(a))

device = torch.device("cpu")
torch.manual_seed(int(time.time()))

kwargs = {'dtype': torch.float64, # TODO: parameterize this later! 
          'device': device,
          'requires_grad': True}

batch_size = 1
n_heads = 4
n_ctx = 4
width = 4
denom = 16 # denominator for random fractions [0 .. 1)

q = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
k = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)

q.retain_grad()
k.retain_grad()
variables = [q, k]

attn = L1AttnFunction.apply(*variables)
d_attn = torch.randint(0, denom, (batch_size, n_heads, n_ctx, n_ctx), **kwargs)/denom
# d_attn = torch.ones(batch_size, n_heads, n_ctx, n_ctx, **kwargs)
attn.backward(d_attn)

# detach tensors for execution graph, clone to cuda
q_cuda = q.detach().clone().cuda()
k_cuda = k.detach().clone().cuda()
d_attn_cuda = d_attn.detach().clone().cuda()

# turn gradient tracing back on
q_cuda.requires_grad = True
k_cuda.requires_grad = True
q_cuda.retain_grad()
k_cuda.retain_grad()
variables_cuda = [q_cuda, k_cuda]

attn_cuda = cuda.l1attn.L1AttnFunction.apply(*variables_cuda)
attn_cuda.backward(d_attn_cuda)

print("grad_q, absum( Naive - Cuda )", absum(q.grad - q_cuda.grad.cpu()).item())
print("grad_k, absum( Naive - Cuda )", absum(k.grad - k_cuda.grad.cpu()).item())

# for manual inspection
if(False):
    print("---")
    print(q.grad)
    print(q_cuda.grad.cpu())
    print("---")
    print(k.grad)
    print(k_cuda.grad.cpu())
