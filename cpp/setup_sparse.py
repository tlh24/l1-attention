from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='l1attnSparse_cpp',
    ext_modules=[
        CppExtension('l1attnSparse_cpp', ['l1attn_sparse.cpp']), #extra_compile_args = [""],
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
    )
