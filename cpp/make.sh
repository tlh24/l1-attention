#!/usr/bin/sh
rm -rf build dist *.egg-info
rm -rf $(pip cache dir)

pip uninstall l1attn-cpp
pip uninstall l1attn-sparse-cpp

python setup.py install
python setup_sparse.py install
