from torch.utils.cpp_extension import load
lltm_cpp = load(name="l1attn_cpp", sources=["l1attn.cpp"], verbose=True)
help(l1attn_cpp)
