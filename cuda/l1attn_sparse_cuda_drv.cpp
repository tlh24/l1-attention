#include <torch/extension.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> l1attnSparse_cuda_forward(
	torch::Tensor v,
	torch::Tensor q,
	torch::Tensor k,
	torch::Tensor coo,
	int dst_mxlen );

std::vector<torch::Tensor> l1attnSparse_cuda_backward(
	torch::Tensor dvo,
	torch::Tensor v,
	torch::Tensor q,
	torch::Tensor k,
	torch::Tensor coo,
	torch::Tensor attn,
	int dst_mxlen );

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> l1attnSparse_forward(
		torch::Tensor v,
		torch::Tensor q,
		torch::Tensor k,
		torch::Tensor coo,
		int dst_mxlen ) 
{
	CHECK_INPUT(v);
	CHECK_INPUT(q);
	CHECK_INPUT(k);
	CHECK_INPUT(coo);

	return l1attnSparse_cuda_forward(v, q, k, coo, dst_mxlen);
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
	CHECK_INPUT(dvo);
	CHECK_INPUT(v);
	CHECK_INPUT(q);
	CHECK_INPUT(k);
	CHECK_INPUT(coo);
	CHECK_INPUT(attn);

	// std::cout << "l1attn_cuda intermediate tensor c" << std::endl;
	// std::cout << c << std::endl;

	return l1attnSparse_cuda_backward(dvo, v, q, k, coo, attn, dst_mxlen); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &l1attnSparse_forward, "L1AttnSparse forward (CUDA)");
  m.def("backward", &l1attnSparse_backward, "L1AttnSparse backward (CUDA)");
}
