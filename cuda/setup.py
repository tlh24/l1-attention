from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='l1attn_cuda',
    ext_modules=[
        CUDAExtension('l1attn_cuda', [
            'l1attn_cuda.cpp',
            'l1attn_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
