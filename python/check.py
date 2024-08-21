import time
import torch
from torch.autograd import gradcheck
from l1attn_sparse import sparseNonsparseTest, LinFun, L1AttnSparse, L1AttnSparseBidi, L1AttnSparseFn, L1AttnSparseBidiFn, expandCoo

if True:
    batch_size = 2
    n_heads = 3
    n_ctx = 4
    width = 5
else: # debug.
    batch_size = 1
    n_heads = 1
    n_ctx = 3
    width = 2

device = torch.device("cpu")
torch.manual_seed(int(time.time()))

kwargs = {'dtype': torch.float64, 
          'device': device,
          'requires_grad': True}

x = torch.randn(batch_size, 3, **kwargs)
w = torch.randn(batch_size, 3, 2, **kwargs)
variables = [x, w]
if gradcheck(LinFun.apply, variables):
    print('Backward: LinFun grad Ok')

# sparseNonsparseTest(True) # from l1attn_sparse.py

q = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
k = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
v = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)

co = torch.cartesian_prod(torch.arange(n_ctx), torch.arange(n_ctx))
coo, dst_mxlen, src_mxlen = expandCoo(co)

m1 = L1AttnSparse()
x1 = m1.forward(v, q, k, coo, dst_mxlen, src_mxlen)
x2 = L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen, src_mxlen)
# print(x1)
# print(x2)
assert( torch.allclose(x1, x2) )

variables = [v, q, k, coo, dst_mxlen, src_mxlen]
if gradcheck(L1AttnSparseFn.apply, variables):
    print('Backward: Baseline grad Ok')

# attention is indexing permutation invariant; check this .
indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

variables = [v, q, k, coo, dst_mxlen, src_mxlen]
if gradcheck(L1AttnSparseFn.apply, variables):
    print('Backward: Baseline grad Ok w/ permutation')

# now drop 4 indices, see if it still works.
indx = indx[0:-4]
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

variables = [v, q, k, coo, dst_mxlen, src_mxlen]
if gradcheck(L1AttnSparseFn.apply, variables):
    print('Backward: Baseline grad Ok w/ sparsity + permutation')
    
### Bidi attention ###

vf = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
vb = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)

m1 = L1AttnSparseBidi()
x1 = m1(vf, vb, q, k, coo, dst_mxlen, src_mxlen)
	
x2 = L1AttnSparseBidiFn.apply(vf, vb, q, k, coo, dst_mxlen, src_mxlen)

assert(torch.allclose(x1, x2))
print("Bidi: Forward output matches.")

variables = [vf, vb, q, k, coo, dst_mxlen, src_mxlen]
if gradcheck(L1AttnSparseBidiFn.apply, variables):
    print('Backward: Baseline Bidi grad Ok')
    
# attention is indexing permutation invariant; check this .
indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

variables = [vf, vb, q, k, coo, dst_mxlen, src_mxlen]
if gradcheck(L1AttnSparseBidiFn.apply, variables):
    print('Backward: Baseline Bidi grad Ok w/ permutation')
    
# now drop 4 indices, see if it still works.
indx = indx[0:-4]
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

variables = [vf, vb, q, k, coo, dst_mxlen, src_mxlen]
if gradcheck(L1AttnSparseBidiFn.apply, variables):
    print('Backward: Baseline Bidi grad Ok w/ sparsity + permutation')
