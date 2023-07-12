#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__  __forceinline__ scalar_t sign(scalar_t x)
{ 
	scalar_t t = x < 0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

template <typename scalar_t>
__global__ void l1attn_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    q,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    k,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    attn,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    c,
    const scalar_t scale, 
    const int bs, const int n_ctx, const int n_heads, const int width)
{
  __shared__ scalar_t acc[8][32];
  
  int tix = threadIdx.x; // [0 .. 31]. 
  int tiy = threadIdx.y; // [0 .. 7]
  // tix operates within across the width dimension (reduction dim) 
  int indx = threadIdx.y + blockIdx.x * 8; // is this right?
  
  if(indx < bs*n_heads*n_ctx*n_ctx){
    // output indexing: gather (scatter might be more efficient..)
    int j = indx; 
    int t = j % n_ctx; 
    j /= n_ctx; 
    int s = j % n_ctx; 
    j /= n_ctx; 
    int h = j % n_heads; 
    j /= n_heads; 
    int b = j; 
  
    int width32 = (width + 31) / 32; 
    scalar_t f = 0.0; 
    for(int w = 0; w < width32; w++) { 
      int o = w*32+tix; 
      if(o < width)
        f += abs(q[b][t][h][o] - k[b][s][h][o]); // why is this transposed??!
    }
    acc[tiy][tix] = f * scale; // width OK: defaults to zero
    if(tix < 16) { 
      acc[tiy][tix] += acc[tiy][tix + 16];
      __syncthreads(); // why is this needed ??? 
      acc[tiy][tix] += acc[tiy][tix + 8 ];
      __syncthreads(); // threads in a warp should be synchronous.
      acc[tiy][tix] += acc[tiy][tix + 4 ];
      __syncthreads();
      acc[tiy][tix] += acc[tiy][tix + 2 ];
      __syncthreads();
      acc[tiy][tix] += acc[tiy][tix + 1 ];
      __syncthreads();
      if(tix == 0){
        f = acc[tiy][tix]; 
        c[b][h][s][t] = f; 
        attn[b][h][s][t] = 1.0 / (0.001 + f); 
      }
    }
  }
}

template <typename scalar_t>
__global__ void l1attn_cuda_backward_kernel_dr(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    d_attn,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    c,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    d_r,
    const int bs, const int n_ctx, const int n_heads ) 
{
  //pre-calculate d_r -- per-element derivative of attention, 
  // based on saved variable c. 
  // otherwise, we need to calculate several times during reduction.
  int indx = threadIdx.x + blockIdx.x * 256; 
  if(indx < bs*n_heads*n_ctx*n_ctx){
    int j = indx; 
    int t = j % n_ctx; 
    j /= n_ctx; 
    int s = j % n_ctx; 
    j /= n_ctx; 
    int h = j % n_heads; 
    j /= n_heads; 
    int b = j; 
    
    scalar_t f = c[b][h][s][t]; 
    // another transpose here .. why???
    d_r[b][h][t][s] = (-1.0 / ((0.001 + f)*(0.001 + f)))
                       * d_attn[b][h][s][t];
    // d_r[b][h][t][s] = f * d_attn[b][h][s][t]; 
  }
}

template <typename scalar_t>
__global__ void l1attn_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    d_r,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    q_acc,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    k_acc,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    d_q,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
    d_k,
    const scalar_t scale, 
    const int bs, const int n_ctx, const int n_heads, const int width ) 
{
  // reduction (across s and t) has to be done within a thread warp: 
  // can't have different warps write the same memory. 
  // they will interfere / not give the correct answer!
  
  int tix = threadIdx.x; // [0 .. 31]. 
  int tiy = threadIdx.y; // [0 .. 7]
  // tix operates within an n_ctx stride. 
  int indx = threadIdx.y + blockIdx.x * 8; 
  
  __shared__ float acc[8][32];
  
  if(indx < bs*n_ctx*n_heads*width){
    // again, output indexing b/c thread blocks can't overlap writes.
    int j = indx; 
    int w = j % width; 
    j /= width; 
    int h = j % n_heads; 
    j /= n_heads; 
    int s = j % n_ctx; 
    j /= n_ctx; 
    int b = j; 
    
    acc[tiy][tix] = 0.0; 
    scalar_t q_s = q_acc[b][s][h][w]; 
    
    int n_ctx32 = (n_ctx + 31) / 32; 
    for(int k = 0; k < n_ctx32; k++){
      int t = k*32 + tix; 
      if(t < n_ctx){
        scalar_t dr = d_r[b][h][s][t]; 
        scalar_t ws = q_s - k_acc[b][t][h][w]; 
        ws = sign(ws); 
        acc[tiy][tix] += ws * dr * scale; 
      }
    }
    if(tix < 16) { 
		acc[tiy][tix] += acc[tiy][tix + 16];
		__syncthreads(); // why is this needed ??? 
		acc[tiy][tix] += acc[tiy][tix + 8 ];
		__syncthreads(); // threads in a warp should be synchronous.
		acc[tiy][tix] += acc[tiy][tix + 4 ];
		__syncthreads();
		acc[tiy][tix] += acc[tiy][tix + 2 ];
		__syncthreads();
		acc[tiy][tix] += acc[tiy][tix + 1 ];
		__syncthreads();
		if(tix == 0){
        d_q[b][s][h][w] = acc[tiy][0];
      }
	}
	
	//now do the same thing for d_k 
	int t = s; 
	acc[tiy][tix] = 0.0; 
    scalar_t k_s = k_acc[b][t][h][w]; 
    
    for(int k = 0; k < n_ctx32; k++){
      int s = k*32 + tix; // bad memory access - I know. 
      if(s < n_ctx){
        scalar_t dr = d_r[b][h][s][t];
        scalar_t ws = q_acc[b][s][h][w] - k_s; 
        ws = sign(ws); 
        acc[tiy][tix] += -1.0 * ws * dr * scale; 
      }
    }
    if(tix < 16) { 
		acc[tiy][tix] += acc[tiy][tix + 16];
		__syncthreads(); // why is this needed ??? 
		acc[tiy][tix] += acc[tiy][tix + 8 ];
		__syncthreads(); // threads in a warp should be synchronous.
		acc[tiy][tix] += acc[tiy][tix + 4 ];
		__syncthreads();
		acc[tiy][tix] += acc[tiy][tix + 2 ];
		__syncthreads();
		acc[tiy][tix] += acc[tiy][tix + 1 ];
		__syncthreads();
		if(tix == 0){
        d_k[b][t][h][w] = acc[tiy][0];
      }
	}
  } // valid indx
} 

std::vector<torch::Tensor> l1attn_cuda_forward(
    torch::Tensor q,
    torch::Tensor k) {
  
  int bs = q.sizes()[0]; 
  int n_ctx = q.sizes()[1]; 
  int n_heads = q.sizes()[2]; 
  int width = q.sizes()[3];
  
  auto options = torch::TensorOptions()
    .dtype(q.dtype())
    .device(q.device())
    .requires_grad(q.requires_grad()); //better way to do this? 
  
  auto attn = torch::zeros({bs, n_heads, n_ctx, n_ctx}, options); 
  auto c = torch::zeros_like(attn); 
  
  const dim3 dimBlocks(32, 8); // x, y, z
  const int n_elements = bs * n_heads * n_ctx * n_ctx; 
  assert((n_elements % 8) == 0); 
  const int n_blocks = n_elements / 8; 
  
  double scale = 1.0 / sqrt(width); 
    
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "l1attn_cuda_forward_kernel", ([&] {
    l1attn_cuda_forward_kernel<scalar_t><<<n_blocks, dimBlocks>>>(
        q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        c.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        scale, bs, n_ctx, n_heads, width);
  }));
  
  return {attn, c};
}

std::vector<torch::Tensor> l1attn_cuda_backward(
    torch::Tensor d_attn,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor c ) {
  
  int bs = q.sizes()[0]; 
  int n_ctx = q.sizes()[1]; 
  int n_heads = q.sizes()[2]; 
  int width = q.sizes()[3];
  
  double scale = 1.0 / sqrt(width);
  
  auto d_r = torch::zeros_like(d_attn); 
  
  //calculate d_r. 
  const int threads = 256; 
  const int n_elements = bs * n_heads * n_ctx * n_ctx; 
  int n_blocks = (n_elements + threads - 1) / threads;
  
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "l1attn_cuda_backward_kernel_dr", ([&] {
    l1attn_cuda_backward_kernel_dr<scalar_t><<<n_blocks, threads>>>(
        d_attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        c.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        d_r.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        bs, n_ctx, n_heads);
  }));
  
  // calculate d_q and d_k
  auto d_q = torch::zeros_like(q);
  auto d_k = torch::zeros_like(k);
  
  const dim3 dimBlocks(32, 8); // x, y, z
  assert((n_elements % 8) == 0); 
  n_blocks = n_elements / 8;
  
  AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "l1attn_cuda_backward_kernel", ([&] {
    l1attn_cuda_backward_kernel<scalar_t><<<n_blocks, dimBlocks>>>(
        d_r.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        d_q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        d_k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        scale, bs, n_ctx, n_heads, width);
  }));
  
  return {d_q, d_k};
}
