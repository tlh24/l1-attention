#include <torch/extension.h>

#include <vector>

#define DTYPE double

DTYPE sign(DTYPE x)
{ 
	DTYPE t = x < 0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

std::vector<torch::Tensor> l1attn_forward(
		torch::Tensor q,
		torch::Tensor k) {
	// don't permute the variables like in python. 
	int bs = q.sizes()[0]; 
	int n_ctx = q.sizes()[1]; 
	int n_heads = q.sizes()[2]; 
	int width = q.sizes()[3]; 

	auto scale = -1.0 / sqrt(width); 

	assert(q.device() == torch::kCPU);
	assert(k.device() == torch::kCPU); 

	auto options = torch::TensorOptions()
		.dtype(q.dtype())
		.device(q.device())
		.requires_grad(q.requires_grad()); 
		
	auto attn = torch::zeros({bs, n_ctx, n_ctx, n_heads}, options);

	// do the subtraction and sum in one pass. 
	// loops to make it clean & simple. 
	auto q_acc = q.accessor<DTYPE,4>(); 
	auto k_acc = k.accessor<DTYPE,4>(); 
	auto attn_acc = attn.accessor<DTYPE,4>(); 

	for(int b = 0; b < bs; b++){
		for(int s = 0; s < n_ctx; s++){
			for(int t = 0; t < n_ctx; t++){
				for(int h = 0; h < n_heads; h++){
					DTYPE f = 0.f; 
					for(int w = 0; w < width; w++){
						f += abs(q_acc[b][t][h][w] - k_acc[b][s][h][w]); 
					}
					f *= scale; 
					attn_acc[b][s][t][h] = f; 
				}
			}
		}
	}
	return {attn};
}

std::vector<torch::Tensor> l1attn_backward(
		torch::Tensor d_attn,
		torch::Tensor q,
		torch::Tensor k) {
	
	int bs = q.sizes()[0]; 
	int n_ctx = q.sizes()[1]; 
	int n_heads = q.sizes()[2]; 
	int width = q.sizes()[3]; 
	
	auto scale = -1.0 / sqrt(width); 
	
	auto q_dtype = q.dtype();  // Get the datatype of tensor q
	auto q_device = q.device();  // Get the device of tensor q
	auto d_q = torch::zeros({bs, n_ctx, n_heads, width});
	d_q = d_q.to(q_device, q_dtype);
	auto d_k = torch::zeros({bs, n_ctx, n_heads, width});
	d_k = d_k.to(q_device, q_dtype);
	
	auto q_acc = q.accessor<DTYPE,4>(); 
	auto k_acc = k.accessor<DTYPE,4>();
	auto d_q_acc = d_q.accessor<DTYPE,4>(); 
	auto d_k_acc = d_k.accessor<DTYPE,4>();
	auto d_attn_acc = d_attn.accessor<DTYPE,4>(); 
	
	for(int b = 0; b < bs; b++){
		for(int s = 0; s < n_ctx; s++){
			for(int t = 0; t < n_ctx; t++){
				for(int h = 0; h < n_heads; h++){
					DTYPE d_a = d_attn_acc[b][s][t][h]; 
					for(int w = 0; w < width; w++){
						DTYPE ws = q_acc[b][t][h][w] - k_acc[b][s][h][w]; 
						ws = sign(ws) * scale; 
						d_q_acc[b][t][h][w] += ws * d_a; 
						d_k_acc[b][s][h][w] -= ws * d_a; 
					}
				}
			}
		}
	}
	return {d_q, d_k}; 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &l1attn_forward, "L1Attn forward");
  m.def("backward", &l1attn_backward, "L1Attn backward");
}
