#!/usr/bin/sh
rm -rf build dist *.egg-info
rm -rf $(pip cache dir)

# Backup the original gcc and g++ symlinks
sudo mv /usr/bin/gcc /usr/bin/gcc.bak
sudo mv /usr/bin/g++ /usr/bin/g++.bak

# Create new symlinks pointing to gcc-12 and g++-12
sudo ln -s /usr/bin/gcc-12 /usr/bin/gcc
sudo ln -s /usr/bin/g++-12 /usr/bin/g++

# Get the site-packages directory
SITE_PACKAGES_DIR=$(python -c "import site; print(site.getsitepackages()[0])")

# Remove any residual files
rm -rf "$SITE_PACKAGES_DIR/l1attn_cuda"*
rm -rf "$SITE_PACKAGES_DIR/l1attn_sparse_cuda"*

pip uninstall l1attn-cuda
pip uninstall l1attn-sparse-cuda # !! you need to remove both !! 

python setup.py install
python setup_sparse.py install

# Restore the original gcc and g++ symlinks
sudo mv -f /usr/bin/gcc.bak /usr/bin/gcc
sudo mv -f /usr/bin/g++.bak /usr/bin/g++
