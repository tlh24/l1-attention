import time
import torch
from torch.autograd import gradcheck
from python.l1attn_sparse import sparseNonsparseTest, LinFun, L1AttnSparse, L1AttnSparseFn, expandCoo
import l1attn_sparse_cpp
import l1attn_sparse_cuda

debug = True

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

device = torch.device("cpu")
torch.manual_seed(int(time.time()))

kwargs = {'dtype': torch.float64, # get errors with float32.
          'device': device,
          'requires_grad': True}

x = torch.randn(batch_size, 3, **kwargs)
w = torch.randn(batch_size, 3, 2, **kwargs)
variables = [x, w]
if gradcheck(LinFun.apply, variables):
	print('Backward: LinFun grad Ok')

sparseNonsparseTest() # from l1attn_sparse.py

q = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
k = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
v = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)

co = torch.tensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
coo, dst_mxlen, src_mxlen = expandCoo(co)

## -- python implementation -- ##

# non-sparse verification.
m1 = L1AttnSparse()
x1 = m1.forward(v, q, k, coo, dst_mxlen, src_mxlen)
x2 = L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen, src_mxlen)
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
    
## -- cpp implementation -- ##

# non-sparse verification.
coo, dst_mxlen, src_mxlen = expandCoo(co)
x3 = l1attn_sparse_cpp.L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen)
assert( torch.allclose(x1, x3) )
print('Forward: Cpp dense Ok')

indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
x3 = l1attn_sparse_cpp.L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen)
assert( torch.allclose(x1, x3) )
print('Forward: Cpp dense, permuted Ok')
    
variables = [v, q, k, coo, dst_mxlen]
if gradcheck(l1attn_sparse_cpp.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
    print('Backward: Cpp grad Ok')

# attention is indexing permutation invariant; check this .
indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

variables = [v, q, k, coo, dst_mxlen]
if gradcheck(l1attn_sparse_cpp.L1AttnSparseFn.apply, variables):
	print('Backward: Cpp grad Ok w/ permutation')

# now drop 4 indices, see if it still works.
indx = indx[0:-4]
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

variables = [v, q, k, coo, dst_mxlen]
if gradcheck(l1attn_sparse_cpp.L1AttnSparseFn.apply, variables):
	print('Backward: Cpp grad Ok w/ sparsity + permutation')

## -- CUDA implementation -- ##

cdevice = torch.device("cuda")
v = v.to(cdevice)
q = q.to(cdevice)
k = k.to(cdevice)

# non-sparse verification.
coo, dst_mxlen, src_mxlen = expandCoo(co)
coo = coo.to(cdevice)
x3 = l1attn_sparse_cuda.L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen)
x3 = x3.to(device)
assert( torch.allclose(x1, x3) )
print('Forward: CUDA dense Ok')

indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
coo = coo.to(cdevice)
x3 = l1attn_sparse_cuda.L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen)
x3 = x3.to(device)
assert( torch.allclose(x1, x3) )
print('Forward: CUDA dense, permuted Ok')

variables = [v, q, k, coo, dst_mxlen]
if gradcheck(l1attn_sparse_cuda.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
    print('Backward: CUDA dense grad Ok')

# attention is indexing permutation invariant; check this .
indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
coo = coo.to(cdevice)

variables = [v, q, k, coo, dst_mxlen]
if gradcheck(l1attn_sparse_cuda.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
	print('Backward: CUDA grad Ok w/ permutation')

# now drop 4 indices, see if it still works.
indx = indx[0:-4]
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
coo = coo.to(cdevice)

variables = [v, q, k, coo, dst_mxlen]
if gradcheck(l1attn_sparse_cuda.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
	print('Backward: CUDA grad Ok w/ sparsity + permutation')
