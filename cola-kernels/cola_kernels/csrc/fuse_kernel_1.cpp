#include <torch/extension.h>

namespace cola_kernels {

at::Tensor fuse_kernel_1_cpu(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.sizes() == c.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(c.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(c.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor c_contig = c.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  const float* c_ptr = c_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c_ptr[i];
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(cola_kernels, m) {
  m.def("fuse_kernel_1(Tensor a, Tensor b, Tensor c) -> Tensor");
}

TORCH_LIBRARY_IMPL(cola_kernels, CPU, m) {
  m.impl("fuse_kernel_1", &fuse_kernel_1_cpu);
}

}