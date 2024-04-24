# torch must be imported first for shared objects to be found properly
import torch
import l1attn

m = l1attn.L1Attn()

# batch_size 2, context 3, heads 4, channels 8
q = torch.randn(2,3,4,8).cuda()
k = torch.randn(2,3,4,8).cuda()

a = m.forward(q,k)
