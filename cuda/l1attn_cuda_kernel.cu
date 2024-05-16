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
__global__ void l1attn_cuda_forward_kernelX(
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
	int h = threadIdx.y; // n_heads
	// tix operates within across the width dimension (reduction dim) 
	int t = blockIdx.x; 
	int s = blockIdx.y; 
	int b = blockIdx.z; 
	
	/*
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
		int b = j; */
	
	int width32 = (width + 31) / 32; 
	scalar_t f = 0.0; 
	for(int w = 0; w < width32; w++) { 
		int o = w*32+tix; 
		if(o < width)
			f += abs(q[b][t][h][o] - k[b][s][h][o]); 
	}
	acc[h][tix] = f * scale; 
	if(tix < 16) { 
		acc[h][tix] += acc[h][tix + 16];
		__syncthreads(); // why is this needed ??? 
		acc[h][tix] += acc[h][tix + 8 ];
		__syncthreads(); // threads in a warp should be synchronous.
		acc[h][tix] += acc[h][tix + 4 ];
		__syncthreads(); // experiment: it's totally needed! 
		acc[h][tix] += acc[h][tix + 2 ];
		__syncthreads();
		acc[h][tix] += acc[h][tix + 1 ];
		__syncthreads();
		if(tix == 0){
			attn[b][s][t][h] = acc[h][tix]; 
		}
	}
}

// template <typename scalar_t>
// __global__ void l1attn_cuda_backward_kernel_old(
// 		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
// 		d_attn,
// 		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
// 		q,
// 		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
// 		k,
// 		torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
// 		d_q,
// 		torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
// 		d_k,
// 		const scalar_t scale, 
// 		const int bs, const int n_ctx, const int n_heads, const int width ) 
// {
// 	// reduction (across s and t) has to be done within a thread warp: 
// 	// can't have different warps write the same memory. 
// 	// they will interfere / not give the correct answer!
// 	
// 	int indx = threadIdx.x + blockIdx.x * blockDim.x; // 1D
// 	
// 	if(indx < bs*n_ctx*n_ctx*n_heads){
// 		// again, output indexing b/c thread blocks can't overlap writes.
// 		// see note in forward kernel.
// 		int j = indx; 
// 		int h = j % n_heads; 
// 		j /= n_heads; 
// 		int s = j % n_ctx; 
// 		j /= n_ctx; 
// 		int t = j % n_ctx; 
// 		j /= n_ctx; 
// 		int b = j % bs; 
// 		
// 		scalar_t d_a = d_attn[b][s][t][h]; 
// 		for(int w = 0; w < width; w++){
// 			scalar_t ws = q[b][t][h][w] - k[b][s][h][w];
// 			ws = sign(ws) * scale; 
// 			// atomicAdd((scalar_t*)&(d_q[b][t][h][w]), ws * d_a);
// 			// atomicAdd((scalar_t*)&(d_k[b][s][h][w]), -1*ws * d_a);
// 			fastAtomicAdd2(d_q, b,t,h,w, ws * d_a);
// 			fastAtomicAdd2(d_k, b,s,h,w, -1*ws * d_a);
// 		}
// 	}
// } 

template <typename scalar_t>
__global__ void l1attn_cuda_backward_kernel(
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		d_attnq,
		const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
		d_attnk,
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
	
	// int tix = threadIdx.x; // [0 .. 31]. 
	// int tiy = threadIdx.y; // [0 .. 7]
	// // tix operates within across the width dimension (reduction dim) 
	// int indx = threadIdx.y + blockIdx.x * 8; // is this right?
	
	int tix = threadIdx.x; // [0 .. 31].
	int h = threadIdx.y; // n_heads
	int u = blockIdx.x; // u is t for q, s for k.
	int w = blockIdx.y; 
	int b = blockIdx.z; 
	
	// if(indx < bs*n_ctx*n_heads*width){
	// 	// again, output indexing b/c thread blocks can't overlap writes.
	// 	// see note in forward kernel.
	// 	int j = indx; 
	// 	int u = j % n_ctx; 
	// 	j /= n_ctx; 
	// 	int h = j % n_heads; 
	// 	j /= n_heads; 
	// 	int w = j % width; 
	// 	j /= width; 
	// 	int b = j % bs; 
		
	int ctx32 = (n_ctx + 31) / 32; 
	scalar_t dq = 0.0; 
	scalar_t dk = 0.0; 
	
	scalar_t qq = q[b][w][h][u]; 
	for(int o = 0; o < ctx32; o++) { 
		int s = o*32+tix; 
		if(s < n_ctx){ 
			// all this would work better if n_ctx were a multiple of 32. 
			scalar_t ws = qq - k[b][w][h][s];
			ws = sign(ws) * scale; 
			scalar_t d_a = d_attnq[b][u][h][s]; 
			dq += ws * d_a; 
		}
	}
	
	scalar_t kk = k[b][w][h][u]; 
	for(int o = 0; o < ctx32; o++) { 
		int t = o*32+tix; 
		if(t < n_ctx){
			scalar_t ws = q[b][w][h][t] - kk;
			ws = sign(ws) * scale; 
			scalar_t d_a = d_attnk[b][u][h][t]; 
			dk -= ws * d_a; 
		}
	}
	
	acc_dq[h][tix] = dq;
	acc_dk[h][tix] = dk;
	if(tix < 16) { 
		acc_dq[h][tix] += acc_dq[h][tix + 16];
		acc_dk[h][tix] += acc_dk[h][tix + 16];
		__syncthreads(); 
		acc_dq[h][tix] += acc_dq[h][tix + 8 ];
		acc_dk[h][tix] += acc_dk[h][tix + 8 ];
		__syncthreads(); 
		acc_dq[h][tix] += acc_dq[h][tix + 4 ];
		acc_dk[h][tix] += acc_dk[h][tix + 4 ];
		__syncthreads();
		acc_dq[h][tix] += acc_dq[h][tix + 2 ];
		acc_dk[h][tix] += acc_dk[h][tix + 2 ];
		__syncthreads();
		acc_dq[h][tix] += acc_dq[h][tix + 1 ];
		acc_dk[h][tix] += acc_dk[h][tix + 1 ];
		__syncthreads();
		if(tix == 0){
			d_q[b][u][h][w] = acc_dq[h][tix];
			d_k[b][u][h][w] = acc_dk[h][tix]; 
		}
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
	
	const dim3 numBlocks(n_ctx, n_ctx, bs); // x, y, z
	// const int n_elements = bs * n_heads * n_ctx * n_ctx; 
	// int n_blocks = (n_elements + 7) / 8;
	// int n_blocks = n_elements;
	// int n_threads = 32;
	const dim3 threadsPerBlock(32, n_heads, 1);
	
	double scale = -1.0 / sqrt(width); 
		
	AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "l1attn_cuda_forward_kernel", ([&] {
		l1attn_cuda_forward_kernelX<scalar_t><<<numBlocks, threadsPerBlock>>>(
			q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			scale, bs, n_ctx, n_heads, width);
	}));
	
	return {attn};
}

std::vector<torch::Tensor> l1attn_cuda_backward(
		torch::Tensor d_attnq,
		torch::Tensor d_attnk,
		torch::Tensor q,
		torch::Tensor k) 
{
	int bs = q.sizes()[0]; // permuted in python driver!!!
	int width = q.sizes()[1];
	int n_heads = q.sizes()[2]; 
	int n_ctx = q.sizes()[3]; 
	
	double scale = -1.0 / sqrt(width);
	
	auto options = torch::TensorOptions()
		.dtype(q.dtype())
		.device(q.device())
		.requires_grad(q.requires_grad());
	
	auto d_q = torch::zeros({bs, n_ctx, n_heads, width}, options);
	auto d_k = torch::zeros({bs, n_ctx, n_heads, width}, options);
	
	// const dim3 dimBlocks(32, 8); // x, y, z
	const dim3 numBlocks(n_ctx, width, bs); // x, y, z
	const dim3 threadsPerBlock(32, n_heads, 1);
	// const int n_elements = bs * n_heads * n_ctx * width; 
	// int n_blocks = n_elements;
	// int n_threads = 32;
	
	AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "l1attn_cuda_backward_kernel", ([&] {
		l1attn_cuda_backward_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
			d_attnq.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			d_attnk.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			d_q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			d_k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			scale, bs, n_ctx, n_heads, width);
	}));
	
	return {d_q, d_k};
}
