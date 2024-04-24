#!/usr/bin/sh
pip uninstall l1attn-cpp
pip uninstall l1attn-sparse-cpp
rm -rf build dist *.egg-info
python setup.py install
python setup_sparse.py install
