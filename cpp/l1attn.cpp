#include <torch/extension.h>

#include <vector>

#define DTYPE double

std::vector<torch::Tensor> l1attn_forward(
    torch::Tensor q,
    torch::Tensor k) {
  // don't permute the variables like in python. 
  int bs = q.sizes()[0]; 
  int n_ctx = q.sizes()[1]; 
  int n_heads = q.sizes()[2]; 
  int width = q.sizes()[3]; 
  
  auto scale = 1.0 / sqrt(width); 
  
  assert(q.device() == torch::kCPU);
  assert(k.device() == torch::kCPU); 
  
  auto options = torch::TensorOptions()
    .dtype(q.dtype())
    .device(q.device())
    .requires_grad(q.requires_grad()); 
    
  auto attn = torch::zeros({bs, n_heads, n_ctx, n_ctx}, options);
  auto c = torch::zeros({bs, n_heads, n_ctx, n_ctx}, options);
  
  // do the subtraction and sum in one pass. 
  // loops to make it clean & simple. 
  auto q_acc = q.accessor<DTYPE,4>(); 
  auto k_acc = k.accessor<DTYPE,4>(); 
  auto attn_acc = attn.accessor<DTYPE,4>(); 
  auto c_acc = c.accessor<DTYPE,4>(); 
  
  for(int b = 0; b < bs; b++){
    for(int s = 0; s < n_ctx; s++){
      for(int t = 0; t < n_ctx; t++){
        for(int h = 0; h < n_heads; h++){
          DTYPE f = 0.f; 
          for(int w = 0; w < width; w++){
            f += abs(q_acc[b][s][h][w] - k_acc[b][t][h][w]); 
          }
          f *= scale; 
          c_acc[b][h][s][t] = f; // save for derivative calc
          f = 1.0 / (0.001 + f); 
          attn_acc[b][h][s][t] = f; 
        }
      }
    }
  }
  return {attn, c};
}

std::vector<torch::Tensor> l1attn_backward(
    torch::Tensor d_attn,
    torch::Tensor q,
    torch::Tensor k, 
    torch::Tensor c) {
  
  int bs = q.sizes()[0]; 
  int n_ctx = q.sizes()[1]; 
  int n_heads = q.sizes()[2]; 
  int width = q.sizes()[3]; 
  
  auto scale = 1.0 / sqrt(width); 
  
  auto q_dtype = q.dtype();  // Get the datatype of tensor q
  auto q_device = q.device();  // Get the device of tensor q
  auto d_q = torch::zeros({bs, n_ctx, n_heads, width});
  d_q = d_q.to(q_device, q_dtype);
  auto d_k = torch::zeros({bs, n_ctx, n_heads, width});
  d_k = d_k.to(q_device, q_dtype);
  
  auto q_acc = q.accessor<DTYPE,4>(); 
  auto k_acc = k.accessor<DTYPE,4>();
  auto c_acc = c.accessor<DTYPE,4>(); 
  auto d_q_acc = d_q.accessor<DTYPE,4>(); 
  auto d_k_acc = d_k.accessor<DTYPE,4>();
  auto d_attn_acc = d_attn.accessor<DTYPE,4>(); 
  
  for(int b = 0; b < bs; b++){
    for(int s = 0; s < n_ctx; s++){
      for(int t = 0; t < n_ctx; t++){
        for(int h = 0; h < n_heads; h++){
          DTYPE f = c_acc[b][h][s][t]; 
          DTYPE d_r = (-1.0 / ((0.001 + f)*(0.001 + f))) 
                      * d_attn_acc[b][h][s][t]; 
          
          for(int w = 0; w < width; w++){
            DTYPE ws = q_acc[b][s][h][w] - k_acc[b][t][h][w]; 
            ws = sign(ws) * scale; 
            d_q_acc[b][s][h][w] += ws * d_r; 
            d_k_acc[b][t][h][w] += ws * d_r * -1.0; 
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
