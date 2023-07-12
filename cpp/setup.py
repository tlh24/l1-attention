from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='l1attn_cpp',
    ext_modules=[
        CppExtension('l1attn_cpp', ['l1attn.cpp']), #extra_compile_args = [""],
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
    )
