#!/usr/bin/sh
rm -rf build dist *.egg-info

pip uninstall l1attn-cuda
python setup.py install

pip uninstall l1attn-sparse-cuda
python setup_sparse.py install
