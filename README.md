# L1 norm attention C++/CUDA Extensions for PyTorch

This is a small library that implements L1 attention via Pytorch extensions in C++ and CUDA.
That is,
```math
Attention[..,i,j] = F( \sum_k abs(Query[..,i,k] - Key[..,j,k]) ) 
```
```math
F(x) = \frac{1}{0.001 + x} 
```
Where the $Attention$ tensor has the leading batch and head dimensions omitted ('..'), $k$ indexes over the per-head channels, and F is applied per-element.  This is pre-softmax.  

In comparison, the more typical dot-product attention: 
```math
Attention[..,i,j] = \sum_k (Query[..,i,k] * Key[..,j,k]) ) 
```
roughly measures the cosine distance between vectors, because of the pre- or post- LayerNorm  (see the [Magneto paper](http://arxiv.org/abs/2210.06423) for further discussion).  (Roughly, depending on where the LayerNorm is and if there are sub-norms.)

L1 attention can solve some symmetric problems that dot-product attention cannot -- see test_transformer.py for an illustration.  It may also solve some problems faster than dot-product attention; see [example](http://github.com/tlh24/l1-attention/tree/main/example) 

This needs to be a C++/ CUDA extension because the form of eq.1 requires the creation of a large tensor, typically sized $[batch_size, num_heads, context_size, context_size, channels], which is then summed over the last dimension.  For large models with large numbers of channels, this both exceeds GPU memory and is grossly inefficient.  

In comparison, the CUDA implementation operates on the Q,K tensors in-place, avoiding allocations other than keeping around one attention-sized tensor for gradient computation.  In the current implementation this is not ''quite'' optimized: to simplify the memory model, each warp (group of 32 GPU threads) only writes one memory location.  That is, it gathers from potentially incoherent memory addresses, and writes coherently.  It would be faster (and more complex) to do the opposite (scatter instead of gather); even better to do blocked attention & include Key operations, ala FlashAttention. 

This library is based on the [Pytorch example](https://github.com/pytorch/extension-cpp). See
[here](http://pytorch.org/tutorials/advanced/cpp_extension.html) for the accompanying tutorial.  It could also be done with [CuPy](https://cupy.dev/), should someone be so interested (There is enough code that I think the extension makes sense.)

## Getting started

First, install `python3-dev` and `nvidia-cuda-toolkit`.  As of March 2024, the module installs properly on Debian testing with pytorch 2.2.1, Cuda 12.0 -- but you need to edit one file in pybind to get it to work, see https://github.com/pybind/pybind11/issues/4606

(Conda ships linked against an old version of gcc -- could not get it to work.)

(Alternately, the CUDA version shipping in Debian testing as of January 2024 (12.0) has a bug when compiled with gcc / g++ 12+ -- and g++11 is no longer in the distro tree.  I've gotten around this by installing Cuda 12.3 from the .run file)

Then, navigate to the `cpp/` and `cuda/` directories and run `python setup.py install` there. 
This will install the l1attn modules into your Conda environment. 

Test the install by running `python check.py` from the root directory of this repository. This will check the forward and backward passes of the L1attn modules against a naive implementation + autograd.  

If all looks good, you may use the CUDA version in your project like this: 
```
# torch must be imported first for shared objects to be found properly
import torch
import l1attn

m = l1attn.L1Attn()

# batch_size 2, context 3, heads 4, channels 8
q = torch.randn(2,3,4,8).cuda()
k = torch.randn(2,3,4,8).cuda()

a = m.forward(q,k)
```
##


## TODO
- Setup / DistUtils will be deprecated in python 3.12.  I'm not an expert here, so will deal with it when needed. 
- Optimize memory access patterns
- Pull in Value computation

## Authors

[Tim Hanson](https://github.com/tlh24)

original example by:
[Peter Goldsborough](https://github.com/goldsborough)

