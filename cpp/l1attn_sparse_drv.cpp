#include <torch/extension.h>

#include <vector>

#define DTYPE double

DTYPE sign(DTYPE x)
{ 
	DTYPE t = x < 0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

std::vector<torch::Tensor> l1attnSparse_forward(
		torch::Tensor v,
		torch::Tensor q,
		torch::Tensor k,
		torch::Tensor coo,
		int dst_mxlen ) 
{
	// don't permute the variables like in python. 
	int bs = q.sizes()[0]; 
	int n_tok = q.sizes()[1];
	int n_heads = q.sizes()[2]; 
	int width = q.sizes()[3]; 
	int cl = coo.sizes()[0];

	auto scale = -1.0 / sqrt(width); 

	assert(v.device() == torch::kCPU);
	assert(q.device() == torch::kCPU);
	assert(k.device() == torch::kCPU); 
	assert(coo.device() == torch::kCPU);

	auto options = torch::TensorOptions()
		.dtype(q.dtype())
		.device(q.device())
		.requires_grad(q.requires_grad()); 
		
	auto attn = torch::ones({bs, n_tok, dst_mxlen, n_heads}, options);
	attn = attn * -1e12; // -infty
	auto vo = torch::zeros({bs, n_tok, n_heads, width}, options);

	auto coo_acc = coo.accessor<int,2>();
	auto v_acc = v.accessor<DTYPE,4>();
	auto q_acc = q.accessor<DTYPE,4>(); 
	auto k_acc = k.accessor<DTYPE,4>(); 
	auto attn_acc = attn.accessor<DTYPE,4>(); // save for backward
	auto vo_acc = vo.accessor<DTYPE,4>();

	for(int b = 0; b < bs; b++){
		for(int h = 0; h < n_heads; h++){
			for(int c = 0; c < cl; c++){
				int dst = coo_acc[c][0];
				int src = coo_acc[c][1];
				int r = coo_acc[c][2]; 
				DTYPE f = 0.0; 
				for(int w = 0; w < width; w++){
					f += abs(q_acc[b][dst][h][w] - k_acc[b][src][h][w]); 
				}
				f *= scale;
				attn_acc[b][dst][r][h] = f; 
			}
			// compute the softmax
			for(int d = 0; d < n_tok; d++){
				DTYPE f = 1e-12; 
				for(int r = 0; r < dst_mxlen; r++)
					f += exp(attn_acc[b][d][r][h]); 
				for(int r = 0; r < dst_mxlen; r++)
					attn_acc[b][d][r][h] = exp(attn_acc[b][d][r][h]) / f; 
			}
			// compute vo
			for(int c = 0; c < cl; c++){
				int dst = coo_acc[c][0];
				int src = coo_acc[c][1];
				int r = coo_acc[c][2];
				for(int w = 0; w < width; w++){
					vo_acc[b][dst][h][w] += 
						attn_acc[b][dst][r][h] * v_acc[b][src][h][w]; 
				}
			}
		}
	}
	return {vo, attn}; 
}
	
std::vector<torch::Tensor> l1attnSparse_backward(
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
	int cl = coo.sizes()[0];

	auto scale = -1.0 / sqrt(width); 
	
	auto options = torch::TensorOptions()
		.dtype(q.dtype())
		.device(q.device())
		.requires_grad(q.requires_grad()); 
	
	auto d_v = torch::zeros({bs, n_tok, n_heads, width}, options);
	auto d_q = torch::zeros({bs, n_tok, n_heads, width}, options);
	auto d_k = torch::zeros({bs, n_tok, n_heads, width}, options);
	auto dattn_sm = torch::zeros({bs, n_tok, dst_mxlen, n_heads}, options);
	auto dattn = torch::zeros({bs, n_tok, dst_mxlen, n_heads}, options);

	auto dvo_acc = dvo.accessor<DTYPE,4>(); 
	auto v_acc = v.accessor<DTYPE,4>(); 
	auto q_acc = q.accessor<DTYPE,4>(); 
	auto k_acc = k.accessor<DTYPE,4>();
	auto coo_acc = coo.accessor<int,2>(); 
	auto attn_acc = attn.accessor<DTYPE,4>(); 
	
	auto dv_acc = d_v.accessor<DTYPE,4>(); 
	auto dq_acc = d_q.accessor<DTYPE,4>(); 
	auto dk_acc = d_k.accessor<DTYPE,4>();
	auto dattn_sm_acc = dattn_sm.accessor<DTYPE,4>();
	auto dattn_acc = dattn.accessor<DTYPE,4>();

	for(int b = 0; b < bs; b++){
		for(int h = 0; h < n_heads; h++){
			for(int c = 0; c < cl; c++){
				int dst = coo_acc[c][0];
				int src = coo_acc[c][1];
				int r = coo_acc[c][2];
				// calc dv -- scale dvo by attn
				for(int w = 0; w < width; w++){
					dv_acc[b][src][h][w] += 
						attn_acc[b][dst][r][h] * dvo_acc[b][dst][h][w]; 
				}
				// calculate dattn pre-softmax. 
				for(int w = 0; w < width; w++){
					dattn_sm_acc[b][dst][r][h] += 
						v_acc[b][src][h][w] * dvo_acc[b][dst][h][w];
				}
			}
			// multiply dattn_sm by jacobian (without allocating it)
			for(int d = 0; d < n_tok; d++){
				for(int r = 0; r < dst_mxlen; r++){
					for(int q = 0; q < dst_mxlen; q++){
						DTYPE f = 0.0; 
						if(r == q)
							f = attn_acc[b][d][r][h] * (1-attn_acc[b][d][r][h]); 
						else
							f = -1 * attn_acc[b][d][r][h] * attn_acc[b][d][q][h]; 
						dattn_acc[b][d][r][h] += f * dattn_sm_acc[b][d][q][h];
					}
				}
			}
			// calculate dq and dk
			for(int c = 0; c < cl; c++){
				int dst = coo_acc[c][0];
				int src = coo_acc[c][1];
				int r = coo_acc[c][2];
				for(int w = 0; w < width; w++){
					DTYPE ws = q_acc[b][dst][h][w] - k_acc[b][src][h][w];
					ws = sign(ws) * scale; 
					dq_acc[b][dst][h][w] += ws * dattn_acc[b][dst][r][h]; 
					dk_acc[b][src][h][w] += -1*ws * dattn_acc[b][dst][r][h]; 
				}
			}
		}
	}
	return {d_v, d_q, d_k};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &l1attnSparse_forward, "L1AttnSparse forward");
  m.def("backward", &l1attnSparse_backward, "L1AttnSparse backward");
}
