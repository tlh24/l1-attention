#!/usr/bin/sh
rm -rf build dist *.egg-info
rm -rf $(pip cache dir)

# Get the site-packages directory
SITE_PACKAGES_DIR=$(python -c "import site; print(site.getsitepackages()[0])")

# Remove any residual files
rm -rf "$SITE_PACKAGES_DIR/l1attn_cpp"*
rm -rf "$SITE_PACKAGES_DIR/l1attn_sparse_cpp"*
rm -rf "$SITE_PACKAGES_DIR/l1attn_sparse_bidi_cpp"*

pip uninstall l1attn-cpp # these will print errors
# pip uninstall l1attn-sparse-cpp # good for pip to note absence?
# pip uninstall l1attn-sparse-bidi-cpp


pip install -e .
wait
