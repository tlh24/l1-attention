#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> l1attnSparseBidi_cuda_forward(
	torch::Tensor vf,
	torch::Tensor vb,
	torch::Tensor q,
	torch::Tensor k,
	torch::Tensor coo,
	int dst_mxlen, 
	bool use_softmax );

std::vector<torch::Tensor> l1attnSparseBidi_cuda_backward(
	torch::Tensor dvo,
	torch::Tensor vf,
	torch::Tensor vb,
	torch::Tensor q,
	torch::Tensor k,
	torch::Tensor coo,
	torch::Tensor attn,
	int dst_mxlen, 
	bool use_softmax );

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> l1attnSparseBidi_forward(
		torch::Tensor vf,
		torch::Tensor vb,
		torch::Tensor q,
		torch::Tensor k,
		torch::Tensor coo,
		int dst_mxlen, 
		bool use_softmax) 
{
	CHECK_INPUT(vf);
	CHECK_INPUT(vb);
	CHECK_INPUT(q);
	CHECK_INPUT(k);
	CHECK_INPUT(coo);

	return l1attnSparseBidi_cuda_forward(vf, vb, q, k, coo, dst_mxlen, use_softmax);
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
	CHECK_INPUT(dvo);
	CHECK_INPUT(vf);
	CHECK_INPUT(vb);
	CHECK_INPUT(q);
	CHECK_INPUT(k);
	CHECK_INPUT(coo);
	CHECK_INPUT(attn);

	// std::cout << "l1attn_cuda intermediate tensor c" << std::endl;
	// std::cout << c << std::endl;

	return l1attnSparseBidi_cuda_backward(dvo, vf, vb, q, k, coo, attn, dst_mxlen, use_softmax); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &l1attnSparseBidi_forward, "L1AttnSparse forward (CUDA)");
  m.def("backward", &l1attnSparseBidi_backward, "L1AttnSparse backward (CUDA)");
}
