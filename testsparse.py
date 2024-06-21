import torch
import l1attn_cuda
import l1attn_sparse_cuda
import torch.nn.functional as F
import pdb

device = torch.device("cuda")

batch_size = 1
n_ctx = 3
n_heads = 2
width = 2

kwargs = {'dtype': torch.float64, 
          'device': device,
          'requires_grad': True}

          
q = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
k = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
v = torch.randn(batch_size, n_ctx, n_heads, width, **kwargs)
          
co = torch.tensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
coo, dst_mxlen, src_mxlen = l1attn_sparse_cuda.expandCoo(co)
coo = coo.to(device)
m = l1attn_sparse_cuda.L1AttnSparse()
vo = m(v, q, k, coo, dst_mxlen, True)
print('ok')

# q = torch.randn(1,3,1,4).cuda()
# k = torch.randn(1,3,1,4).cuda()
# v = torch.randn(1,3,1,4).cuda() 

q = torch.zeros(1,3,1,4).cuda()
k = torch.zeros(1,3,1,4).cuda()
v = torch.zeros(1,3,1,4).cuda() 

q[0,0,0,:] = torch.tensor([0,1,0,0])
q[0,1,0,:] = torch.tensor([0,0,1,0])
q[0,2,0,:] = torch.tensor([1,0,0,0])

k[0,0,0,:] = torch.tensor([1,0,0,0])
k[0,1,0,:] = torch.tensor([0,1,0,0])
k[0,2,0,:] = torch.tensor([0,0,1,0])

v[0,0,0,:] = torch.tensor([1,0,0,1])
v[0,1,0,:] = torch.tensor([0,1,0,2])
v[0,2,0,:] = torch.tensor([0,0,1,3])

md = l1attn_cuda.L1Attn()
a = md.forward(q,k)
a = torch.cat((a, torch.zeros(1,1,3,1).cuda()), axis=1)
a_sm = F.softmax(a, 1)
a_sm = a_sm[:,:-1,:,:]
vo = torch.einsum('bsdh, bshw -> bdhw', a_sm, v)
print(vo)

co = torch.tensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
coo, dst_mxlen, src_mxlen = l1attn_sparse_cuda.expandCoo(co)
coo = coo.to(device)
vo = m(v, q, k, coo, dst_mxlen, True)
print(vo)
