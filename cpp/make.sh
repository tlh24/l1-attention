#!/usr/bin/sh
rm -rf build dist *.egg-info

pip uninstall l1attn-cpp
python setup.py install

pip uninstall l1attn-sparse-cpp
python setup_sparse.py install
