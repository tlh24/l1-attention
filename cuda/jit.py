from torch.utils.cpp_extension import load
lltm_cuda = load(
    'l1attn_cuda', ['l1attn_cuda.cpp', 'l1attn_cuda_kernel.cu'], verbose=True)
help(l1attn_cuda)
