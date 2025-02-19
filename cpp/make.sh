#!/usr/bin/sh
rm -rf build dist *.egg-info
rm -rf $(pip cache dir)

# Get the site-packages directory
SITE_PACKAGES_DIR=$(python -c "import site; print(site.getsitepackages()[0])")

# Remove any residual files
rm -rf "$SITE_PACKAGES_DIR/l1attn_cpp"*
rm -rf "$SITE_PACKAGES_DIR/l1attn_sparse_cpp"*
rm -rf "$SITE_PACKAGES_DIR/l1attn_sparse_bidi_cpp"*

echo "no build isolation: be sure to have setuptools, wheel and torch installed>
pip install --no-build-isolation -e .
wait


echo "Checking installed modules..."
python -c "
import importlib.util
modules = ['l1attn_cpp', 'l1attn_sparse_cpp', 'l1attn_sparse_bidi_cpp', 
           'l1attn_drv_cpp', 'l1attn_sparse_drv', 'l1attn_sparse_bidi_drv']
for module in modules:
    try:
        __import__(module)
        print(f'{module}: Installed')
    except ImportError:
        print(f'{module}: Not installed')
"
