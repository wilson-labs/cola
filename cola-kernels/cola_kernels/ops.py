import torch
from torch import Tensor

__all__ = ["fuse_kernel_1"]


def fuse_kernel_1(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.cola_kernels.fuse_kernel_1.default(a, b, c)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("cola_kernels::fuse_kernel_1")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.shape == c.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(c.dtype == torch.float)
    torch._check(a.device == b.device)
    torch._check(a.device == c.device)
    return torch.empty_like(a)
