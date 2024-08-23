import time
import torch
from torch.autograd import gradcheck
from python.l1attn_sparse import sparseNonsparseTest, LinFun, L1AttnSparse, L1AttnSparseFn, L1AttnSparseBidi, expandCoo
import l1attn_sparse_cpp
import l1attn_sparse_bidi_cpp
import l1attn_sparse_cuda
import l1attn_sparse_bidi_cuda

debug = False

if debug:
	batch_size = 1
	n_ctx = 3
	n_heads = 2
	width = 2
else: 
	batch_size = 2
	n_ctx = 3
	n_heads = 16
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

sparseNonsparseTest(True) # from l1attn_sparse.py
sparseNonsparseTest(False) # from l1attn_sparse.py

q = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
k = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
v = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)

co = torch.cartesian_prod(torch.arange(n_ctx), torch.arange(n_ctx))
coo, dst_mxlen, src_mxlen = expandCoo(co)

## -- python implementation -- ##

# non-sparse verification.
m1 = L1AttnSparse()
for use_softmax in [True, False]: 
	x1 = m1.forward(v, q, k, coo, dst_mxlen, src_mxlen, use_softmax)
	x2 = L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen, src_mxlen, use_softmax)
	assert( torch.allclose(x1, x2) )

	variables = [v, q, k, coo, dst_mxlen, src_mxlen, use_softmax]
	if gradcheck(L1AttnSparseFn.apply, variables):
		print(f'Backward: Baseline grad Ok use_softmax={use_softmax}')

# attention is indexing permutation invariant; check this .
indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

for use_softmax in [True, False]: 
	variables = [v, q, k, coo, dst_mxlen, src_mxlen, use_softmax]
	if gradcheck(L1AttnSparseFn.apply, variables):
		print(f'Backward: Baseline grad Ok w/ permutation, use_softmax={use_softmax}')

# now drop 4 indices, see if it still works.
indx = indx[0:-4]
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

for use_softmax in [True, False]: 
	variables = [v, q, k, coo, dst_mxlen, src_mxlen, use_softmax]
	if gradcheck(L1AttnSparseFn.apply, variables):
		print(f'Backward: Baseline grad Ok w/ sparsity + permutation, use_softmax={use_softmax}')
    
## -- cpp implementation -- ##

# non-sparse verification.
coo, dst_mxlen, src_mxlen = expandCoo(co)
for use_softmax in [True, False]: 
	x1 = m1.forward(v, q, k, coo, dst_mxlen, src_mxlen, use_softmax)
	x3 = l1attn_sparse_cpp.L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen, use_softmax)
	assert( torch.allclose(x1, x3) )
	print(f'Forward: Cpp Ok, use_softmax={use_softmax}')

indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
for use_softmax in [True, False]: 
	x1 = m1.forward(v, q, k, coo, dst_mxlen, src_mxlen, use_softmax)
	x3 = l1attn_sparse_cpp.L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen, use_softmax)
	assert( torch.allclose(x1, x3) )
	print(f'Forward: Cpp, permuted Ok, use_softmax={use_softmax}')
  
coo, dst_mxlen, src_mxlen = expandCoo(co)
for use_softmax in [True, False]: 
	variables = [v, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_cpp.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
		print(f'Backward: Cpp grad Ok, use_softmax={use_softmax}')

# attention is indexing permutation invariant; check this .
indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

for use_softmax in [True, False]: 
	variables = [v, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_cpp.L1AttnSparseFn.apply, variables):
		print(f'Backward: Cpp grad Ok w/ permutation, use_softmax={use_softmax}')

# now drop 4 indices, see if it still works.
indx = indx[0:-4]
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

for use_softmax in [True, False]: 
	variables = [v, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_cpp.L1AttnSparseFn.apply, variables):
		print(f'Backward: Cpp grad Ok w/ sparsity + permutation, sm={use_softmax}')
		
# -- Bidi implementation -- #

vf = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
vb = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
m1b = L1AttnSparseBidi()

# non-sparse verification.
coo, dst_mxlen, src_mxlen = expandCoo(co)
for use_softmax in [True, False]: 
	x1 = m1b.forward(vf, vb, q, k, coo, dst_mxlen, src_mxlen, use_softmax)
	x3 = l1attn_sparse_bidi_cpp.L1AttnSparseBidiFn.apply(vf, vb, q, k, coo, dst_mxlen, use_softmax)
	assert( torch.allclose(x1, x3) )
	print(f'Forward: Cpp bidi Ok, use_softmax={use_softmax}')

indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
for use_softmax in [True, False]: 
	x1 = m1b.forward(vf, vb, q, k, coo, dst_mxlen, src_mxlen, use_softmax)
	x3 = l1attn_sparse_bidi_cpp.L1AttnSparseBidiFn.apply(vf, vb, q, k, coo, dst_mxlen, use_softmax)
	assert( torch.allclose(x1, x3) )
	print(f'Forward: Cpp bidi, permuted Ok, use_softmax={use_softmax}')
  
coo, dst_mxlen, src_mxlen = expandCoo(co)
for use_softmax in [True, False]: 
	variables = [vf, vb, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_bidi_cpp.L1AttnSparseBidiFn.apply, variables, nondet_tol=1e-6):
		print(f'Backward: Cpp bidi grad Ok, use_softmax={use_softmax}')

# attention is indexing permutation invariant; check this .
indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

for use_softmax in [True, False]: 
	variables = [vf, vb, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_bidi_cpp.L1AttnSparseBidiFn.apply, variables):
		print(f'Backward: Cpp bidi grad Ok w/ permutation, use_softmax={use_softmax}')

# now drop 4 indices, see if it still works.
indx = indx[0:-4]
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)

for use_softmax in [True, False]: 
	variables = [vf, vb, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_bidi_cpp.L1AttnSparseBidiFn.apply, variables):
		print(f'Backward: Cpp bidi grad Ok w/ sparsity + permutation, use_softmax={use_softmax}')


## -- CUDA implementation -- ##


cdevice = torch.device("cuda")
v = v.to(cdevice)
q = q.to(cdevice)
k = k.to(cdevice)
# non-sparse verification.
coo, dst_mxlen, src_mxlen = expandCoo(co)
coo = coo.to(cdevice)
for use_softmax in [True, False]: 
	x1 = m1.forward(v, q, k, coo, dst_mxlen, src_mxlen, use_softmax)
	x3 = l1attn_sparse_cuda.L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen, use_softmax)
	assert( torch.allclose(x1, x3) )
	print(f'Forward: Cuda dense Ok, use_softmax={use_softmax}')

indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
coo = coo.to(cdevice)
for use_softmax in [True, False]: 
	x1 = m1.forward(v, q, k, coo, dst_mxlen, src_mxlen, use_softmax)
	x3 = l1attn_sparse_cuda.L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen, use_softmax)
	assert( torch.allclose(x1, x3) )
	print(f'Forward: Cuda dense, permuted Ok, use_softmax={use_softmax}')
  
coo, dst_mxlen, src_mxlen = expandCoo(co)
coo = coo.to(cdevice)
for use_softmax in [True, False]: 
	variables = [v, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_cuda.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
		print(f'Backward: Cuda grad Ok, use_softmax={use_softmax}')

# attention is indexing permutation invariant; check this .
indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
coo = coo.to(cdevice)

for use_softmax in [True, False]: 
	variables = [v, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_cuda.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
		print(f'Backward: Cuda grad Ok w/ permutation, use_softmax={use_softmax}')

# now drop 4 indices, see if it still works.
indx = indx[0:-4]
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
coo = coo.to(cdevice)

for use_softmax in [True, False]: 
	variables = [v, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_cuda.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
		print(f'Backward: Cuda grad Ok w/ sparsity + permutation, sm={use_softmax}')

# -- Bidi Cuda implementation -- #

cdevice = torch.device("cuda")
vf = vf.to(cdevice)
vb = vb.to(cdevice)

# non-sparse verification.
coo, dst_mxlen, src_mxlen = expandCoo(co)
coo = coo.to(cdevice)
for use_softmax in [True, False]: 
	x1 = m1b.forward(vf, vb, q, k, coo, dst_mxlen, src_mxlen, use_softmax)
	x3 = l1attn_sparse_bidi_cuda.L1AttnSparseBidiFn.apply(vf, vb, q, k, coo, dst_mxlen, use_softmax)
	assert( torch.allclose(x1, x3) )
	print(f'Forward: Cuda bidi Ok, use_softmax={use_softmax}')

indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
coo = coo.to(cdevice)
for use_softmax in [True, False]: 
	x1 = m1b.forward(vf, vb, q, k, coo, dst_mxlen, src_mxlen, use_softmax)
	x3 = l1attn_sparse_bidi_cuda.L1AttnSparseBidiFn.apply(vf, vb, q, k, coo, dst_mxlen, use_softmax)
	assert( torch.allclose(x1, x3) )
	print(f'Forward: Cuda bidi, permuted Ok, use_softmax={use_softmax}')
  
coo, dst_mxlen, src_mxlen = expandCoo(co)
coo = coo.to(cdevice)
for use_softmax in [True, False]: 
	variables = [vf, vb, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_bidi_cuda.L1AttnSparseBidiFn.apply, variables, nondet_tol=1e-6):
		print(f'Backward: Cuda bidi grad Ok, use_softmax={use_softmax}')

# attention is indexing permutation invariant; check this.
indx = torch.randperm(co.shape[0])
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
coo = coo.to(cdevice)

for use_softmax in [True, False]: 
	variables = [vf, vb, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_bidi_cuda.L1AttnSparseBidiFn.apply, variables, nondet_tol=1e-6):
		print(f'Backward: Cuda bidi grad Ok w/ permutation, use_softmax={use_softmax}')

# now drop 4 indices, see if it still works.
indx = indx[0:-4]
co2 = co[indx, :]
coo, dst_mxlen, src_mxlen = expandCoo(co2)
coo = coo.to(cdevice)

for use_softmax in [True, False]: 
	variables = [vf, vb, q, k, coo, dst_mxlen, use_softmax]
	if gradcheck(l1attn_sparse_bidi_cuda.L1AttnSparseBidiFn.apply, variables, nondet_tol=1e-6):
		print(f'Backward: Cuda bidi grad Ok w/ sparsity + permutation, sm={use_softmax}')


# 
# 
# # non-sparse verification.
# coo, dst_mxlen, src_mxlen = expandCoo(co)
# coo = coo.to(cdevice)
# x3 = l1attn_sparse_cuda.L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen)
# x3 = x3.to(device)
# assert( torch.allclose(x1, x3) )
# print('Forward: CUDA dense Ok')
# 
# indx = torch.randperm(co.shape[0])
# co2 = co[indx, :]
# coo, dst_mxlen, src_mxlen = expandCoo(co2)
# coo = coo.to(cdevice)
# x3 = l1attn_sparse_cuda.L1AttnSparseFn.apply(v, q, k, coo, dst_mxlen)
# x3 = x3.to(device)
# assert( torch.allclose(x1, x3) )
# print('Forward: CUDA dense, permuted Ok')
# 
# variables = [v, q, k, coo, dst_mxlen]
# if gradcheck(l1attn_sparse_cuda.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
#     print('Backward: CUDA dense grad Ok')
# 
# # attention is indexing permutation invariant; check this .
# indx = torch.randperm(co.shape[0])
# co2 = co[indx, :]
# coo, dst_mxlen, src_mxlen = expandCoo(co2)
# coo = coo.to(cdevice)
# 
# variables = [v, q, k, coo, dst_mxlen]
# if gradcheck(l1attn_sparse_cuda.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
# 	print('Backward: CUDA grad Ok w/ permutation')
# 
# # now drop 4 indices, see if it still works.
# indx = indx[0:-4]
# co2 = co[indx, :]
# coo, dst_mxlen, src_mxlen = expandCoo(co2)
# coo = coo.to(cdevice)
# 
# variables = [v, q, k, coo, dst_mxlen]
# if gradcheck(l1attn_sparse_cuda.L1AttnSparseFn.apply, variables, nondet_tol=1e-6):
# 	print('Backward: CUDA grad Ok w/ sparsity + permutation')
