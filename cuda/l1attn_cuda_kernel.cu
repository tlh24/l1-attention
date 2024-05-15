#include <torch/extension.h>
#include <ATen/native/cuda/KernelUtils.cuh>

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
__device__  __forceinline__ void fastAtomicAdd2(
	torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> out, 
	int i0, int i1, int i2, int i3, scalar_t v)
{
	// convenience wrapper function around
	// fastAtomicAdd for 4-D tensors. 
	int index = i0*out.stride(0) + i1*out.stride(1) + i2*out.stride(2) + i3*out.stride(3);
	at::native::fastAtomicAdd(out.data(), index, 1, v, true); 
}

template <typename scalar_t>
__global__ void l1attn_cuda_forward_kernel(
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		q,
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		k,
		torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		attn,
		const scalar_t scale, 
		const int bs, const int n_ctx, const int n_heads, const int width)
{
	__shared__ scalar_t acc[8][32];
	
	int tix = threadIdx.x; // [0 .. 31]. 
	int tiy = threadIdx.y; // [0 .. 7]
	// tix operates within across the width dimension (reduction dim) 
	int indx = threadIdx.y + blockIdx.x * 8; // is this right?
	
	if(indx < bs*n_heads*n_ctx*n_ctx){
		// we can permute the order of the output indexing here to improve
		// memory gather coherency.  
		// but, because each warp can only write one mem loc, 
		// it's still a gather operation.
		// empirical notes: permuting the indexing order did not change speed! 
		int j = indx; 
		int h = j % n_heads; 
		j /= n_heads; 
		int t = j % n_ctx; 
		j /= n_ctx; 
		int s = j % n_ctx; 
		j /= n_ctx; 
		int b = j; 
	
		int width32 = (width + 31) / 32; 
		scalar_t f = 0.0; 
		for(int w = 0; w < width32; w++) { 
			int o = w*32+tix; 
			if(o < width)
			f += abs(q[b][t][h][o] - k[b][s][h][o]); 
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
				attn[b][s][t][h] = acc[tiy][tix]; 
			}
		}
	}
}

template <typename scalar_t>
__global__ void l1attn_cuda_backward_kernel_old(
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		d_attn,
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		q,
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		k,
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
	
	int indx = threadIdx.x + blockIdx.x * blockDim.x; // 1D
	
	if(indx < bs*n_ctx*n_ctx*n_heads){
		// again, output indexing b/c thread blocks can't overlap writes.
		// see note in forward kernel.
		int j = indx; 
		int h = j % n_heads; 
		j /= n_heads; 
		int s = j % n_ctx; 
		j /= n_ctx; 
		int t = j % n_ctx; 
		j /= n_ctx; 
		int b = j % bs; 
		
		scalar_t d_a = d_attn[b][s][t][h]; 
		for(int w = 0; w < width; w++){
			scalar_t ws = q[b][t][h][w] - k[b][s][h][w];
			ws = sign(ws) * scale; 
			// atomicAdd((scalar_t*)&(d_q[b][t][h][w]), ws * d_a);
			// atomicAdd((scalar_t*)&(d_k[b][s][h][w]), -1*ws * d_a);
			fastAtomicAdd2(d_q, b,t,h,w, ws * d_a);
			fastAtomicAdd2(d_k, b,s,h,w, -1*ws * d_a);
		}
	}
} 

template <typename scalar_t>
__global__ void l1attn_cuda_backward_kernel(
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		d_attn,
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		q,
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		k,
		torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		d_q,
		torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		d_k,
		const scalar_t scale, 
		const int bs, const int n_ctx, const int n_heads, const int width ) 
{
	__shared__ scalar_t acc_dq[8][32];
	__shared__ scalar_t acc_dk[8][32];
	
	int tix = threadIdx.x; // [0 .. 31]. 
	int tiy = threadIdx.y; // [0 .. 7]
	// tix operates within across the width dimension (reduction dim) 
	int indx = threadIdx.y + blockIdx.x * 8; // is this right?
	
	if(indx < bs*n_ctx*n_heads*width){
		// again, output indexing b/c thread blocks can't overlap writes.
		// see note in forward kernel.
		int j = indx; 
		int w = j % width; 
		j /= width; 
		int h = j % n_heads; 
		j /= n_heads; 
		int u = j % n_ctx; // u is t for q, s for k.
		j /= n_ctx; 
		int b = j % bs; 
		
		int ctx32 = (n_ctx + 31) / 32; 
		scalar_t dq = 0.0; 
		scalar_t dk = 0.0; 
		
		scalar_t qq = q[b][u][h][w]; 
		for(int o = 0; o < ctx32; o++) { 
			int s = o*32+tix; 
			if(s < n_ctx){
				scalar_t ws = qq - k[b][s][h][w];
				ws = sign(ws) * scale; 
				scalar_t d_a = d_attn[b][s][u][h]; 
				dq += ws * d_a; 
			}
		}
		
		scalar_t kk = k[b][u][h][w]; 
		for(int o = 0; o < ctx32; o++) { 
			int t = o*32+tix; 
			if(t < n_ctx){
				scalar_t ws = q[b][t][h][w] - kk;
				ws = sign(ws) * scale; 
				scalar_t d_a = d_attn[b][u][t][h]; 
				dk += -1.0 * ws * d_a; 
			}
		}
		
		acc_dq[tiy][tix] = dq;
		acc_dk[tiy][tix] = dk;
		if(tix < 16) { 
			acc_dq[tiy][tix] += acc_dq[tiy][tix + 16];
			acc_dk[tiy][tix] += acc_dk[tiy][tix + 16];
			__syncthreads(); // why is this needed ??? 
			acc_dq[tiy][tix] += acc_dq[tiy][tix + 8 ];
			acc_dk[tiy][tix] += acc_dk[tiy][tix + 8 ];
			__syncthreads(); // threads in a warp should be synchronous.
			acc_dq[tiy][tix] += acc_dq[tiy][tix + 4 ];
			acc_dk[tiy][tix] += acc_dk[tiy][tix + 4 ];
			__syncthreads();
			acc_dq[tiy][tix] += acc_dq[tiy][tix + 2 ];
			acc_dk[tiy][tix] += acc_dk[tiy][tix + 2 ];
			__syncthreads();
			acc_dq[tiy][tix] += acc_dq[tiy][tix + 1 ];
			acc_dk[tiy][tix] += acc_dk[tiy][tix + 1 ];
			__syncthreads();
			if(tix == 0){
				d_q[b][u][h][w] = acc_dq[tiy][tix];
				d_k[b][u][h][w] = acc_dk[tiy][tix]; 
			}
		}
		
		
		// for(int w = 0; w < width; w++){
		// 	scalar_t ws = q[b][t][h][w] - k[b][s][h][w];
		// 	ws = sign(ws) * scale; 
		// 	// atomicAdd((scalar_t*)&(d_q[b][t][h][w]), ws * d_a);
		// 	// atomicAdd((scalar_t*)&(d_k[b][s][h][w]), -1*ws * d_a);
		// 	fastAtomicAdd2(d_q, b,t,h,w, ws * d_a);
		// 	fastAtomicAdd2(d_k, b,s,h,w, -1*ws * d_a);
		// }
	}
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
	
	auto attn = torch::zeros({bs, n_ctx, n_ctx, n_heads}, options); 
	
	const dim3 dimBlocks(32, 8); // x, y, z
	const int n_elements = bs * n_heads * n_ctx * n_ctx; 
	int n_blocks = (n_elements + 7) / 8;
	
	double scale = -1.0 / sqrt(width); 
		
	AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "l1attn_cuda_forward_kernel", ([&] {
		l1attn_cuda_forward_kernel<scalar_t><<<n_blocks, dimBlocks>>>(
			q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			scale, bs, n_ctx, n_heads, width);
	}));
	
	return {attn};
}

std::vector<torch::Tensor> l1attn_cuda_backward(
		torch::Tensor d_attn,
		torch::Tensor q,
		torch::Tensor k) 
{
	int bs = q.sizes()[0]; 
	int n_ctx = q.sizes()[1]; 
	int n_heads = q.sizes()[2]; 
	int width = q.sizes()[3];
	
	double scale = -1.0 / sqrt(width);
	
	auto d_q = torch::zeros_like(q);
	auto d_k = torch::zeros_like(k);
	
	const dim3 dimBlocks(32, 8); // x, y, z
	const int n_elements = bs * n_heads * n_ctx * width; 
	int n_blocks = (n_elements + 7) / 8;
	
	AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "l1attn_cuda_backward_kernel", ([&] {
		l1attn_cuda_backward_kernel<scalar_t><<<n_blocks, dimBlocks>>>(
			d_attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			d_q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			d_k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			scale, bs, n_ctx, n_heads, width);
	}));
	
	return {d_q, d_k};
}
