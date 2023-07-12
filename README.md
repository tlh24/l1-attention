# L1 norm attention C++/CUDA Extensions for PyTorch

This is a small library that implements L1 attention via Pytorch extensions in C++ and CUDA.
That is,
```math
Attention[..,i,j] = F( \sum_k abs(Query[..,i,k] - Key[..,j,k]) ) 
\\
F(x) = \frac{1}{0.001 + x} 
```
Where the $Attention$ tensor has the leading batch and head dimensions omitted ('..') and $k$ indexes over the per-head channels.  This is pre-softmax.  

In comparison, the more typical dot-product attention: 
```math
Attention[..,i,j] = \sum_k (Query[..,i,k] * Key[..,j,k]) ) 
```
roughly measures the cosine distance between vectors, because of the pre- or post- LayerNorm  (see the [Magneto paper](http://arxiv.org/abs/2210.06423) for further discussion).  (Roughly, depending on where the LayerNorm is and if there are sub-norms.)

L1 attention can solve some symmetric problems that dot-product attention cannot -- see test_transformer.py for an illustration.  It may also solve some problems faster than dot-product attention; however, work is ongoing! 

This needs to be a C++/ CUDA extension because the form of eq.1 requires the creation of a large tensor, typically sized $[batch_size, num_heads, context_size, context_size, channels], which is then summed over the last dimension.  For large models with large numbers of channels, this both exceeds GPU memory and is grossly inefficient.  

In comparison, the CUDA implementation operates on the Q,K tensors in-place, avoiding allocations other than keeping around one attention-sized tensor for gradient computation.  In the current implementation this is not ''quite'' optimized: to simplify the memory model, each warp (group of 32 GPU threads) only writes one memory location.  That is, it gathers from potentially incoherent memory addresses, and writes coherently.  It would be faster (and more complex) to do the opposite (scatter instead of gather); even better to do blocked attention & include Key operations, ala FlashAttention. 

This library is based on the [Pytorch example](https://github.com/pytorch/extension-cpp). See
[here](http://pytorch.org/tutorials/advanced/cpp_extension.html) for the accompanying tutorial.  It could also be done with [CuPy](https://cupy.dev/), should someone be so interested (There is enough code that I think the extension makes sense.)

## Getting started

You may need to install Pytorch from source within a Conda environment to get this working on your hardware; I did (RTX 4090; shader model 8.9; tested against Pytorch 2.1.0a0+gitde7b6e5).  Please refer to the pytorch [documentation](https://github.com/pytorch/pytorch#from-source).  

(As of June 2023, Conda ships linked against an old version of gcc -- much prefer virtual environments, but couldn't get them to work.)

Then, navigate to the `cpp/` and `cuda/` directories and run `python setup.py install` there. 
This will install the l1attn modules into your Conda environment. 

Test the install by running `python check.py` from the root directory of this repository. This will check the forward and backward passes of the L1attn modules against a naive implementation + autograd.  


## Authors

[Tim Hanson](https://github.com/tlh24)

original example by:
[Peter Goldsborough](https://github.com/goldsborough)

