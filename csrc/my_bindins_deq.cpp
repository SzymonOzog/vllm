#include <torch/extension.h>
#include <pybind11/pybind11.h>

torch::Tensor ggml_dequantize(torch::Tensor W,  // quant weight
                              int64_t type, int64_t m, int64_t n, std::optional<at::ScalarType> const& dtype);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_deq", &ggml_dequantize, "ggmldeq", py::arg("W"), py::arg("type"),
        py::arg("m"), py::arg("n"), py::arg("dtype") = py::none());
  //m.def("ggml_deq", &ggml_dequantize, "ggmldeq");
}
