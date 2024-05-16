#!/usr/bin/sh
rm -rf build dist *.egg-info
rm -rf $(pip cache dir)

# Get the site-packages directory
SITE_PACKAGES_DIR=$(python -c "import site; print(site.getsitepackages()[0])")

# Remove any residual files
rm -rf "$SITE_PACKAGES_DIR/l1attn_cuda"*
rm -rf "$SITE_PACKAGES_DIR/l1attn_sparse_cuda"*

pip uninstall l1attn-cuda
pip uninstall l1attn-sparse-cuda # !! you need to remove both !! 

python setup.py install
python setup_sparse.py install
