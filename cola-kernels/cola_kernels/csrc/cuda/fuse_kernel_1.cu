#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace cola_kernels {

__global__ void fuse_kernel_1(int numel, const float* a, const float* b, const float* c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c[idx];
}

at::Tensor fuse_kernel_1_cuda(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.sizes() == c.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(c.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(c.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor c_contig = c.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  const float* c_ptr = c_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  int numel = a_contig.numel();
  fuse_kernel_1<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c_ptr, result_ptr);
  return result;
}

TORCH_LIBRARY_IMPL(cola_kernels, CUDA, m) {
  m.impl("fuse_kernel_1", &fuse_kernel_1_cuda);
}

}

