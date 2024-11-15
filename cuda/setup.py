from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
]

setup(
    name='l1attn_cuda_combined',
    version='0.1.0',
    py_modules=['l1attn_cuda', 'l1attn_sparse_cuda', 'l1attn_sparse_bidi_cuda'],
    ext_modules=[
        CUDAExtension('l1attn_cuda_drv', [
            'l1attn_cuda_drv.cpp',
            'l1attn_cuda_kernel.cu',
        ],
        extra_compile_args={"nvcc": nvcc_flags},
        ),
        CUDAExtension('l1attn_sparse_cuda_drv', [
            'l1attn_sparse_cuda_drv.cpp',
            'l1attn_sparse_cuda_kernel.cu',
        ],
        extra_compile_args={"nvcc": nvcc_flags},
        ),
        CUDAExtension('l1attn_sparse_bidi_cuda_drv', [
            'l1attn_sparse_bidi_cuda_drv.cpp',
            'l1attn_sparse_cuda_kernel.cu'
        ],
        extra_compile_args={"nvcc": nvcc_flags},
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
#  # might be worth adding more flags, e.g. ptxas options
#  # https://github.com/facebookresearch/xformers/blob/main/setup.py
#  # is a suitable refrence. 
 
#  # note: latest Debian gcc is v 13; does not work with nvcc from CUDA 12.1
