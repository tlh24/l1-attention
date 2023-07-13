from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='l1attn_cuda',
    py_modules=['l1attn'],
    ext_modules=[
        CUDAExtension('l1attn_cuda', [
            'l1attn_cuda.cpp',
            'l1attn_cuda_kernel.cu',],
        extra_compile_args={
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v", # verbose
                ]   
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
 # might be worth adding more flags, e.g. ptxas options
 # https://github.com/facebookresearch/xformers/blob/main/setup.py
 # is a suitable refrence. 
