# torch must be imported first for shared objects to be found properly
import torch
import torch.nn.functional as F
import l1attn_cuda
import pdb

m = l1attn_cuda.L1Attn()

# batch_size 2, context 3, heads 4, channels 8
q = torch.randn(2,3,4,8).cuda()
k = torch.randn(2,3,4,8).cuda()
v = torch.randn(2,3,4,8).cuda()

a = m.forward(q,k)
a_sm = F.softmax(a, 1) # compute the softmax
vo = torch.einsum('bsdh, bshw -> bdhw', a_sm, v) # weight value output
print('ok')


q = torch.zeros(1,4,1,4).cuda()
k = torch.zeros(1,4,1,4).cuda()
v = torch.zeros(1,4,1,4).cuda()

q[0,0,0,:] = torch.tensor([0,1,0,0])
q[0,1,0,:] = torch.tensor([0,0,1,0])
q[0,2,0,:] = torch.tensor([1,0,0,0])

k[0,0,0,:] = torch.tensor([1,0,0,0])
k[0,1,0,:] = torch.tensor([0,1,0,0])
k[0,2,0,:] = torch.tensor([0,0,1,0])

v[0,0,0,:] = torch.tensor([1,0,0,1])
v[0,1,0,:] = torch.tensor([0,1,0,2])
v[0,2,0,:] = torch.tensor([0,0,1,3])

pdb.set_trace()
a = m.forward(q,k)
a_sm = F.softmax(a, 1)
vo = torch.einsum('bsdh, bshw -> bdhw', a_sm, v)
print(vo)
