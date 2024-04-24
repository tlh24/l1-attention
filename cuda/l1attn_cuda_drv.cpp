#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> l1attn_cuda_forward(
    torch::Tensor q,
    torch::Tensor k );

std::vector<torch::Tensor> l1attn_cuda_backward(
    torch::Tensor d_attn,
    torch::Tensor q,
    torch::Tensor k);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> l1attn_forward(
		torch::Tensor q,
		torch::Tensor k ) {
	CHECK_INPUT(q);
	CHECK_INPUT(k);

	return l1attn_cuda_forward(q, k);
}

std::vector<torch::Tensor> l1attn_backward(
		torch::Tensor d_attn,
		torch::Tensor q,
		torch::Tensor k) {
	CHECK_INPUT(d_attn);
	CHECK_INPUT(q);
	CHECK_INPUT(k);

	// std::cout << "l1attn_cuda intermediate tensor c" << std::endl;
	// std::cout << c << std::endl;

	return l1attn_cuda_backward(d_attn, q, k); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &l1attn_forward, "L1Attn forward (CUDA)");
  m.def("backward", &l1attn_backward, "L1Attn backward (CUDA)");
}
