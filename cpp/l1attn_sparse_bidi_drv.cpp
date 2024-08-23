#include <torch/extension.h>

#include <vector>

#define DTYPE double

DTYPE sign(DTYPE x)
{ 
	DTYPE t = x < 0 ? -1 : 0;
	return x > 0 ? 1 : t;
}

/* Bidirectional sparse attention */

std::vector<torch::Tensor> l1attnSparseBidi_forward(
		torch::Tensor vf,
		torch::Tensor vb,
		torch::Tensor q,
		torch::Tensor k,
		torch::Tensor coo,
		int dst_mxlen, 
		bool use_softmax) 
{
	// don't permute the variables like in python. 
	int bs = q.sizes()[0]; 
	int n_tok = q.sizes()[1];
	int n_heads = q.sizes()[2]; 
	int width = q.sizes()[3]; 
	int cl = coo.sizes()[0];

	auto scale = -1.0 / sqrt(width); 

	assert(vf.device() == torch::kCPU);
	assert(vb.device() == torch::kCPU);
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
	auto vf_acc = vf.accessor<DTYPE,4>();
	auto vb_acc = vb.accessor<DTYPE,4>();
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
			if(use_softmax){ 
				// compute the softmax
				for(int d = 0; d < n_tok; d++){
					DTYPE f = 1; // denominator bias; was 1e-12;
					for(int r = 0; r < dst_mxlen; r++)
						f += exp(attn_acc[b][d][r][h]); 
					for(int r = 0; r < dst_mxlen; r++)
						attn_acc[b][d][r][h] = exp(attn_acc[b][d][r][h]) / f; 
				}
			} else {
				for(int d = 0; d < n_tok; d++){
					for(int r = 0; r < dst_mxlen; r++)
						attn_acc[b][d][r][h] = exp(attn_acc[b][d][r][h]); 
				}
			}
			// compute vfo & vbo, implicitly
			for(int c = 0; c < cl; c++){
				int dst = coo_acc[c][0];
				int src = coo_acc[c][1];
				int r = coo_acc[c][2];
				for(int w = 0; w < width; w++){ //gather
					vo_acc[b][dst][h][w] += 
						attn_acc[b][dst][r][h] * vf_acc[b][src][h][w]; 
				}
				for(int w = 0; w < width; w++){ //scatter: swap dst,src
					vo_acc[b][src][h][w] += 
						attn_acc[b][dst][r][h] * vb_acc[b][dst][h][w]; 
				}
			}
		}
	}
	return {vo, attn}; 
}
	
std::vector<torch::Tensor> l1attnSparseBidi_backward(
		torch::Tensor dvo,
		torch::Tensor vf,
		torch::Tensor vb,
		torch::Tensor q,
		torch::Tensor k,
		torch::Tensor coo,
		torch::Tensor attn,
		int dst_mxlen,
		bool use_softmax) 
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
	
	auto d_vf = torch::zeros({bs, n_tok, n_heads, width}, options);
	auto d_vb = torch::zeros({bs, n_tok, n_heads, width}, options);
	auto d_q = torch::zeros({bs, n_tok, n_heads, width}, options);
	auto d_k = torch::zeros({bs, n_tok, n_heads, width}, options);
	auto dattn_sm = torch::zeros({bs, n_tok, dst_mxlen, n_heads}, options);
	auto dattn = torch::zeros({bs, n_tok, dst_mxlen, n_heads}, options);

	auto dvo_acc = dvo.accessor<DTYPE,4>(); 
	auto vf_acc = vf.accessor<DTYPE,4>(); 
	auto vb_acc = vb.accessor<DTYPE,4>(); 
	auto q_acc = q.accessor<DTYPE,4>(); 
	auto k_acc = k.accessor<DTYPE,4>();
	auto coo_acc = coo.accessor<int,2>(); 
	auto attn_acc = attn.accessor<DTYPE,4>(); 
	
	auto dvf_acc = d_vf.accessor<DTYPE,4>();
	auto dvb_acc = d_vb.accessor<DTYPE,4>();
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
				// calc dvf -- scale dvo by attn
				for(int w = 0; w < width; w++){
					dvf_acc[b][src][h][w] += 
						attn_acc[b][dst][r][h] * dvo_acc[b][dst][h][w]; 
				}
				// calc dvb -- scale dvo by attn transpose
				for(int w = 0; w < width; w++){
					dvb_acc[b][dst][h][w] += 
						attn_acc[b][dst][r][h] * dvo_acc[b][src][h][w]; 
				}
				// calculate dattn pre-softmax. 
				for(int w = 0; w < width; w++){
					dattn_sm_acc[b][dst][r][h] += 
						vf_acc[b][src][h][w] * dvo_acc[b][dst][h][w];
				}
				for(int w = 0; w < width; w++){
					dattn_sm_acc[b][dst][r][h] += 
						vb_acc[b][dst][h][w] * dvo_acc[b][src][h][w];
				}
			}
			if(use_softmax){
				// multiply dattn_sm by sm jacobian (without allocating it)
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
			} else {
				for(int d = 0; d < n_tok; d++){
					for(int r = 0; r < dst_mxlen; r++){
						dattn_acc[b][d][r][h] += 
							attn_acc[b][d][r][h] * dattn_sm_acc[b][d][r][h];
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
	return {d_vf, d_vb, d_q, d_k};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &l1attnSparseBidi_forward, "L1AttnSparseBidi forward");
  m.def("backward", &l1attnSparseBidi_backward, "L1AttnSparseBidi backward");
}
