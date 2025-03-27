#include <torch/extension.h>
#include <pybind11/pybind11.h>

torch::Tensor ggml_moe_a8(torch::Tensor X, torch::Tensor W,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens);

torch::Tensor ggml_moe_kenel(torch::Tensor X, torch::Tensor W,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_padded, int64_t type,
                             int64_t row, int64_t top_k, int64_t tokens) {
  return ggml_moe_a8(X, W, sorted_token_ids, expert_ids, num_tokens_post_padded,
                     type, row, top_k, tokens);
}

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_vec_a8_k(torch::Tensor W, torch::Tensor X,
                                    int64_t type, int64_t row) {
  return ggml_mul_mat_vec_a8(W, X, type, row);
}

torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);

torch::Tensor ggml_mul_mat_a8_k(torch::Tensor W, torch::Tensor X, int64_t type,
                                int64_t row) {
  return ggml_mul_mat_a8(W, X, type, row);
}

int64_t ggml_moe_get_block_size(int64_t type);

int64_t ggml_moe_get_block_size_k(int64_t type) {
    return ggml_moe_get_block_size(type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_moe_a8", &ggml_moe_kenel, "GGML moe kernel");
  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8_k, "matvedc");
  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8_k, "matvedc");
  m.def("ggml_moe_get_block_size", &ggml_moe_get_block_size_k, "matvedc");
}
