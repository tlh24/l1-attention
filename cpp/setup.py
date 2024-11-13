from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
	name='l1attn_cpp',
	version = '0.1.0',
	py_modules=['l1attn_cpp', 'l1attn_sparse_cpp', 'l1attn_sparse_bidi_cpp'],
	ext_modules=[
		CppExtension('l1attn_drv_cpp', ['l1attn_drv.cpp']),
		CppExtension('l1attn_sparse_drv', ['l1attn_sparse_drv.cpp']),
		CppExtension('l1attn_sparse_bidi_drv', ['l1attn_sparse_bidi_drv.cpp']),
	],
	cmdclass={
		'build_ext': BuildExtension
	}
)
