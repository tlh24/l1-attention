#!/usr/bin/sh
pip uninstall l1attn-cuda
pip uninstall l1attn-sparse-cuda
rm -rf build dist *.egg-info
python setup.py install
python setup_sparse.py install
