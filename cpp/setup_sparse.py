from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='l1attn_sparse_cpp',
    version = '0.1.0',
    py_modules=['l1attn_sparse_cpp'],
    ext_modules=[
        CppExtension('l1attn_sparse_cpp_drv', ['l1attn_sparse_drv.cpp']), #extra_compile_args = [""],
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
