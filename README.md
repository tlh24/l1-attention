# L1 norm attention C++/CUDA Extensions for PyTorch

This is a small library that implements L1 attention via Pytorch extensions in C++ and CUDA.
Query,Key,Value tensors are assumed to be shaped like `[batch_size, n_tokens, heads, width]`.  
Dense attention does the (very simple) sum: 
```math
Attention[b,i,j,h] = - \sum_k abs(Query[b,i,h,k] - Key[b,j,h,k])
```
Attention is therefore the computed as the L1 distance (norm) between queries and keys.  This is pre-softmax - note the negative, which allows you to pass the resulting attention tensor directly to softmax (see example below). 

In comparison, the more typical dot-product attention: 
```math
Attention[..,i,j] = \sum_k (Query[..,i,k] * Key[..,j,k]) ) 
```
roughly measures the cosine distance between vectors, because of the pre- or post- LayerNorm  (see the [Magneto paper](http://arxiv.org/abs/2210.06423) for further discussion).  (Roughly, depending on where the LayerNorm is and if there are sub-norms.)

L1 attention can solve some symmetric problems that dot-product attention cannot -- see test_transformer.py for an illustration.  It may also solve some problems faster than dot-product attention; see [example](http://github.com/tlh24/l1-attention/tree/main/example) 

This needs to be a C++/ CUDA extension because the form of eq.1 requires the creation of a large tensor, typically sized $[batch_size, num_heads, n_tokens, n_tokens, width], which is then summed over the last dimension.  For large models with large numbers of channels, this both exceeds GPU memory and is grossly inefficient.  

In comparison, the CUDA implementation operates on the Q,K tensors in-place, avoiding allocations other than keeping around one attention-sized tensor for gradient computation.  In the current implementation this is not ''quite'' optimized: to simplify the memory model, each warp (group of 32 GPU threads) only writes one memory location.  That is, it gathers from potentially incoherent memory addresses, and writes coherently.  It would be faster (and more complex) to do the opposite (scatter instead of gather); even better to do blocked attention & include Key operations, ala FlashAttention. 

This library is based on the [Pytorch example](https://github.com/pytorch/extension-cpp). See
[here](http://pytorch.org/tutorials/advanced/cpp_extension.html) for the accompanying tutorial.  It could also be done with [CuPy](https://cupy.dev/), should someone be so interested (There is enough code that I think the extension makes sense.)

# Sparse L1 attention

Since writing the library, we've become interested in sparse transformers -- not blocked sparsity, but fully flexible sparsity.  This is useful for e.g. doing operations on a graph.  Rather than specifying a mask over an attention, here we use a 'coordinate vector' in the form of a list of `[dest,src]` rows, where `dest` and `src` index tokens. 

Each row in the coordinate vector specifies a query (indexed by `dest`) that is compared via L1 distance to a key (indexed by `src`).  These distances is collated, softmax is taken over the `src` dimension for each unique `dest`, and this is used to form a weighted sum of the values.  

It's verified that this is equivalent to dense attention (above), and is independent of coordinate vector permutations.  

A basic CUDA implementation is provided, but be forewarned -- I resorted to a few atomicAdd operations to avoid warp indexing gymnastics!

## Getting started

First, install `python3-dev` and `nvidia-cuda-toolkit`, then make a virtual environment (if you do not already have one), and install pytorch.  As of March 2024, the module installs properly on Debian testing with pytorch 2.2.1, Cuda 12.0 or 12.3 -- but you need to edit one file in pybind to get it to work, see https://github.com/pybind/pybind11/issues/4606

(Conda ships linked against an old version of gcc -- could not get it to work.)

(Likewise, the CUDA version shipping in Debian testing as of January 2024 (12.0) has a bug when compiled with gcc / g++ 12+ -- and g++11 is no longer in the distro tree.  I've gotten around this by installing Cuda 12.3 from the .run file w/ gcc-12.  See [bug](https://github.com/pybind/pybind11/issues/4606) )


Then, navigate to the `cpp/` and `cuda/` directories and run `make.sh` there. 
This will install the l1attn modules into your virtual environment. 

Test the install by running `python check.py` and `python check_sparse.py` from the root directory of this repository. This will check the forward and backward passes of the L1attn modules against a naive implementation + autograd.  

If all looks good, you may use the dense CUDA version in your project like this: 
```
# torch must be imported first for shared objects to be found properly
import torch
import torch.nn.functional as F
import l1attn_cuda

m = l1attn_cuda.L1Attn()

# batch_size 2, context 3, heads 4, channels 8
q = torch.randn(2,3,4,8).cuda()
k = torch.randn(2,3,4,8).cuda()
v = torch.randn(2,3,4,8).cuda()

a = m.forward(q,k)
a_sm = F.softmax(a, 1) # compute the softmax
vo = torch.einsum('bsdh, bshw -> bdhw', a_sm, v) # weight value output
```
To use sparse attention, you need to know the coordinate vector `coo`: 
```
import torch
import l1attn_sparse_cuda

device = torch.device("cuda")

batch_size = 1
n_ctx = 3
n_heads = 2
width = 2

kwargs = {'dtype': torch.float64, 
          'device': device,
          'requires_grad': True}

          
q = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
k = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
v = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
          
co = torch.tensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
coo, dst_mxlen, src_mxlen = l1attn_sparse_cuda.expandCoo(co)
coo = coo.to(device)
m = l1attn_sparse_cuda.L1AttnSparse()
vo = m(v, q, k, coo, dst_mxlen)
print('ok')


```


## TODO
- ~~Setup / DistUtils will be deprecated in python 3.12.  I'm not an expert here, so will deal with it when needed~~  Fixed in pull request [7](https://github.com/tlh24/l1-attention/pull/7)
- Optimize memory access patterns
- Merge the 4 packages into one, based on where the tensors are (Cpu or Cuda) and if they are dense or sparse.  

## Authors

[Tim Hanson](https://github.com/tlh24)

original example by:
[Peter Goldsborough](https://github.com/goldsborough)

