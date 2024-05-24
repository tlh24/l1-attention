#include <torch/extension.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__  __forceinline__ scalar_t sign(scalar_t x)
{ 
	scalar_t t = x < 0 ? -1 : 0;
	return x > 0 ? (scalar_t)1 : t;
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
__global__ void l1attnSparse_fwd_attn_kernel(
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	q,
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	k,
	const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits>
	coo,
	torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	attn,
	const scalar_t scale, 
	const int bs, const int n_tok, const int n_heads, const int width, 
	const int cl, const int dst_mxlen)
{
	__shared__ scalar_t acc[8][32];

	int tix = threadIdx.x; // [0 .. 31]. 
	int tiy = threadIdx.y; // [0 .. 7]
	// tix operates within the width dimension (reduction dim) 
	// so that threads in a warp have coherent mem access
	int indx = tiy + blockIdx.x * 8;

	if(indx < bs*n_heads*cl){
		int j = indx; 
		int h = j % n_heads; 
		j /= n_heads; 
		int b = j % bs; 
		j /= bs; 
		int c = j % cl; 

		int dst = coo[c][0]; // these must be unique
		int src = coo[c][1]; // ow multiple warps wil write to the same loc
		int r = coo[c][2]; // resulting in nondeterministic behavior
		
		int width32 = (width + 31) / 32; 
		scalar_t f = 0.0; 
		for(int w = 0; w < width32; w++) { 
			int o = w*32+tix; 
			if(o < width)
				f += abs(q[b][dst][h][o] - k[b][src][h][o]);
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
				attn[b][dst][r][h] = acc[tiy][tix]; 
			}
		}
	}
}

template <typename scalar_t>
__global__ void l1attnSparse_fwd_sm_kernel(
	torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	attn,
	const int bs, const int n_tok, const int n_heads, const int width, 
	const int cl, const int dst_mxlen)
{

	int indx = threadIdx.x + blockIdx.x * blockDim.x;

	if(indx < bs*n_heads*n_tok){
		int j = indx; 
		int h = j % n_heads; 
		j /= n_heads; 
		int b = j % bs; 
		j /= bs; 
		int d = j; 
		
		// ideally, each thread in a warp should access consecutive memory. 
		// this is hard since dst_mxlen is potentially small.
		// simple option is to loop over this index within the thread. 
		// this avoids race cases as each thread operates in-place,
		// on one row by itself.
		
		scalar_t f = 1; // denominator; was 1e-12;
		for(int r = 0; r < dst_mxlen; r++){
			f += exp(attn[b][d][r][h]);
		}
		for(int r = 0; r < dst_mxlen; r++){
			attn[b][d][r][h] = exp(attn[b][d][r][h]) / f; 
		}
	}
}

template <typename scalar_t>
__global__ void l1attnSparse_fwd_vo_kernel(
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	v,
	const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits>
	coo,
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits>
	attn,
	torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits>
	vo, 
	const int bs, const int n_tok, const int n_heads, const int v_width, 
	const int cl, const int dst_mxlen)
{
	// 1D threads and blocks.
	int indx = threadIdx.x + blockIdx.x * blockDim.x;

	if(indx < bs*n_heads*cl*v_width){
		int j = indx; 
		int w = j % v_width; 
		j /= v_width; 
		int h = j % n_heads; 
		j /= n_heads; 
		int b = j % bs; 
		j /= bs; 
		int c = j % cl; 

		int dst = coo[c][0]; 
		int src = coo[c][1]; 
		int r = coo[c][2]; 

		// atomicAdd((scalar_t*)&(vo[b][dst][h][w]), attn[b][dst][r][h] * v[b][src][h][w]);
		fastAtomicAdd2(vo, b,dst,h,w, attn[b][dst][r][h] * v[b][src][h][w]);
	}
}

template <typename scalar_t>
__global__ void l1attnSparse_bkwd_dv_dattn_sm_kernel(
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	dvo,
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	v,
	const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits>
	coo,
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits>
	attn,
	torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	dv,
	torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	dattn_sm,
	const int bs, const int n_tok, const int n_heads, const int v_width, 
	const int cl, const int dst_mxlen)
{
	__shared__ scalar_t acc[8][32];
	
	int tix = threadIdx.x; // [0 .. 31]. 
	int tiy = threadIdx.y; // [0 .. 7]
	// tix operates within across the width dimension (reduction dim) 
	int indx = tiy + blockIdx.x * 8;

	if(indx < bs*n_heads*cl){
		int j = indx; 
		int h = j % n_heads; 
		j /= n_heads; 
		int b = j % bs; 
		j /= bs; 
		int c = j % cl; 

		int dst = coo[c][0]; 
		int src = coo[c][1]; 
		int r = coo[c][2]; 
		scalar_t at = attn[b][dst][r][h];
		
		// this is the simplest possible version -- 
		// just use atomic summation to vo, 
		// same as the C++ version.
		// each thread reads attn, dvo & writes dv, dattn_sm
		
		int width32 = (v_width + 31) / 32; 
		scalar_t f = 0.0; 
		for(int w = 0; w < width32; w++) { 
			int o = w*32+tix; 
			if(o < v_width){
				// calc dv
				// atomicAdd((scalar_t*)&(dv[b][src][h][o]), at * dvo[b][dst][h][o]);
				fastAtomicAdd2(dv, b,src,h,o, at * dvo[b][dst][h][o]); 
				// sum dattn_sm
				f += v[b][src][h][o] * dvo[b][dst][h][o];
			}
		}
		acc[tiy][tix] = f; 
		if(tix < 16) {
			acc[tiy][tix] += acc[tiy][tix + 16];
			__syncthreads();
			acc[tiy][tix] += acc[tiy][tix + 8 ];
			__syncthreads();
			acc[tiy][tix] += acc[tiy][tix + 4 ];
			__syncthreads();
			acc[tiy][tix] += acc[tiy][tix + 2 ];
			__syncthreads();
			acc[tiy][tix] += acc[tiy][tix + 1 ];
			__syncthreads();
			if(tix == 0){
				dattn_sm[b][dst][r][h] = acc[tiy][tix]; 
			}
		}
	}
}

template <typename scalar_t>
__global__ void l1attnSparse_bkwd_dattn_kernel(
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	attn,
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	dattn_sm,
	torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	dattn,
	const int bs, const int n_tok, const int n_heads, const int width, 
	const int cl, const int dst_mxlen)
{
	// each thread computes one element of dattn 
	// (dst_mxlen ops)

	int indx = threadIdx.x + blockIdx.x * blockDim.x; // 1D
	
	if(indx < bs*n_heads*n_tok*dst_mxlen){
		int j = indx;
		int h = j % n_heads; 
		j /= n_heads; 
		int r = j % dst_mxlen; 
		j /= dst_mxlen; 
		int b = j % bs; 
		j /= bs; 
		int d = j % n_tok; 
		
		scalar_t acc = 0.0; 
		
		for(int q = 0; q < dst_mxlen; q++){
			scalar_t f = 0.0; 
			if(r == q)
				f = attn[b][d][r][h] * (1-attn[b][d][r][h]);
			else
				f = -1 * attn[b][d][r][h] * attn[b][d][q][h];
			acc += f * dattn_sm[b][d][q][h]; 
		}
		dattn[b][d][r][h] = acc; // one unique write per thread
		// this avoids having to coalesce info from a warp
	}
}
	
template <typename scalar_t>
__global__ void l1attnSparse_bkwd_dq_dk_kernel(
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	q,
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	k,
	const torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits>
	coo,
	const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits>
	dattn,
	torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	dq,
	torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> 
	dk,
	const scalar_t scale, 
	const int bs, const int n_tok, const int n_heads, const int width, 
	const int cl, const int dst_mxlen)
{
	// again, each thread computes & writes one element of dq and dk
	int indx = threadIdx.x + blockIdx.x * blockDim.x; // 1D

	if(indx < width*bs*n_heads*cl){
		// TODO: see if permuting this order speeds things up.
		int j = indx; 
		int w = j % width; 
		j /= width; 
		int h = j % n_heads; 
		j /= n_heads; 
		int b = j % bs; 
		j /= bs; 
		int c = j % cl; 

		int dst = coo[c][0]; 
		int src = coo[c][1]; 
		int r = coo[c][2]; 
		
		scalar_t ws = q[b][dst][h][w] - k[b][src][h][w]; 
		ws = sign(ws) * scale; 
		// atomicAdd((scalar_t*)&(dq[b][dst][h][w]), ws * dattn[b][dst][r][h]);
		// atomicAdd((scalar_t*)&(dk[b][src][h][w]), -1*ws * dattn[b][dst][r][h]);
		fastAtomicAdd2(dq, b,dst,h,w, ws * dattn[b][dst][r][h]); 
		fastAtomicAdd2(dk, b,src,h,w, -1*ws * dattn[b][dst][r][h]); 
	}
}


std::vector<torch::Tensor> l1attnSparse_cuda_forward(
		torch::Tensor v,
		torch::Tensor q,
		torch::Tensor k,
		torch::Tensor coo,
		int dst_mxlen ) 
{
	int bs = q.sizes()[0]; 
	int n_tok = q.sizes()[1];
	int n_heads = q.sizes()[2]; 
	int width = q.sizes()[3];
	int v_width = v.sizes()[3]; 
	int cl = coo.sizes()[0];

	auto options = torch::TensorOptions()
		.dtype(q.dtype())
		.device(q.device())
		.requires_grad(q.requires_grad()); 

	auto attn = torch::ones({bs, n_tok, dst_mxlen, n_heads}, options);
	attn = attn * -1e12; // -infty
	auto vo = torch::zeros({bs, n_tok, n_heads, v_width}, options);

	const dim3 dimBlocks(32, 8); // x, y, z
	int n_elements = bs * n_heads * cl;
	int n_blocks = (n_elements + 7) / 8;

	auto scale = -1.0 / sqrt(width); 
		
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "l1attnSparse_fwd_attn_kernel", ([&] {
		l1attnSparse_fwd_attn_kernel<scalar_t><<<n_blocks, dimBlocks>>>(
			q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			coo.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
			attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			scale, bs, n_tok, n_heads, width, cl, dst_mxlen);
	}));
	
	const int threads = 256; 
	n_elements = bs * n_heads * n_tok; 
	n_blocks = (n_elements + threads - 1) / threads;
	
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "l1attnSparse_fwd_sm_kernel", ([&] {
		l1attnSparse_fwd_sm_kernel<scalar_t><<<n_blocks, threads>>>(
			attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			bs, n_tok, n_heads, width, cl, dst_mxlen);
	}));
	
	n_elements = bs * n_heads * cl * v_width; 
	n_blocks = (n_elements + threads - 1) / threads;
	
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "l1attnSparse_fwd_vo_kernel", ([&] {
		l1attnSparse_fwd_vo_kernel<scalar_t><<<n_blocks, threads>>>(
			v.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			coo.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
			attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			vo.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			bs, n_tok, n_heads, v_width, cl, dst_mxlen);
	}));

	return {vo, attn};
}

std::vector<torch::Tensor> l1attnSparse_cuda_backward(
		torch::Tensor dvo,
		torch::Tensor v,
		torch::Tensor q,
		torch::Tensor k,
		torch::Tensor coo,
		torch::Tensor attn,
		int dst_mxlen ) 
{
	int bs = q.sizes()[0]; 
	int n_tok = q.sizes()[1]; 
	int n_heads = q.sizes()[2]; 
	int width = q.sizes()[3]; 
	int v_width = v.sizes()[3]; 
	int cl = coo.sizes()[0];

	auto scale = -1.0 / sqrt(width); 
	
	auto options = torch::TensorOptions()
		.dtype(q.dtype())
		.device(q.device())
		.requires_grad(q.requires_grad()); 
	
	auto dv = torch::zeros({bs, n_tok, n_heads, v_width}, options);
	auto dq = torch::zeros({bs, n_tok, n_heads, width}, options);
	auto dk = torch::zeros({bs, n_tok, n_heads, width}, options);
	auto dattn_sm = torch::zeros({bs, n_tok, dst_mxlen, n_heads}, options);
	auto dattn = torch::zeros({bs, n_tok, dst_mxlen, n_heads}, options);
	
	const dim3 dimBlocks(32, 8); // x, y, z
	int n_elements = bs * n_heads * cl;
	int n_blocks = (n_elements + 7) / 8;
		
	AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "l1attnSparse_bkwd_dv_dattn_sm_kernel", ([&] {
		l1attnSparse_bkwd_dv_dattn_sm_kernel<scalar_t><<<n_blocks, dimBlocks>>>(
			dvo.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			v.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			coo.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
			attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			dv.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			dattn_sm.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			bs, n_tok, n_heads, v_width, cl, dst_mxlen);
	}));
	
	const int threads = 256; 
	n_elements = bs * n_heads * n_tok * dst_mxlen; 
	n_blocks = (n_elements + threads - 1) / threads;
	
	AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "l1attnSparse_bkwd_dattn_kernel", ([&] {
		l1attnSparse_bkwd_dattn_kernel<scalar_t><<<n_blocks, threads>>>(
			attn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			dattn_sm.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			dattn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			bs, n_tok, n_heads, width, cl, dst_mxlen);
	}));
	
	n_elements = bs * n_heads * cl * width; 
	n_blocks = (n_elements + threads - 1) / threads;
	
	AT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "l1attnSparse_bkwd_dq_dk_kernel", ([&] {
		l1attnSparse_bkwd_dq_dk_kernel<scalar_t><<<n_blocks, threads>>>(
			q.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			k.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			coo.packed_accessor32<int,2,torch::RestrictPtrTraits>(),
			dattn.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			dq.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			dk.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			scale, bs, n_tok, n_heads, width, cl, dst_mxlen);
	}));
	
	return {dv, dq, dk};
}
