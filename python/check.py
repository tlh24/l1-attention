import time
import torch
from torch.autograd import gradcheck
from l1attn_sparse import L1AttnSparse, L1AttnSparseFn, expandCoo


batch_size = 1
n_heads = 2
n_ctx = 3
width = 4

device = torch.device("cpu")
torch.manual_seed(int(time.time()))

kwargs = {'dtype': torch.float64, 
          'device': device,
          'requires_grad': True}

q = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
k = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
v = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)

co = torch.tensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
coo, coo_max_cnt = expandCoo(co)

m1 = L1AttnSparse()
x1 = m1.forward(q, k, v, coo, coo_max_cnt)
x2 = L1AttnSparseFn.apply(q, k, v, coo, coo_max_cnt)
assert( torch.allclose(x1, x2) )

variables = [q, k, v, coo, coo_max_cnt]
if gradcheck(L1AttnSparseFn.apply, variables):
    print('Backward: Baseline grad Ok')
