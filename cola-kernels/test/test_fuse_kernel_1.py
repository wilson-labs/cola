import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import cola_kernels
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F


def reference_fuse_kernel_1(a, b, c):
    return a * b + c


class TestFuseKernel1(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), make_tensor(3)],
            [make_tensor(20), make_tensor(20), make_tensor(20)],
            [make_tensor(20), make_nondiff_tensor(20), make_tensor(20)],
            [make_nondiff_tensor(2, 3), make_tensor(2, 3), make_tensor(2, 3)],
        ]

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = cola_kernels.ops.fuse_kernel_1(*args)
            expected = reference_fuse_kernel_1(*args)
            torch.testing.assert_close(result, expected)

    def test_correctness_cpu(self):
        self._test_correctness("cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")


if __name__ == "__main__":
    unittest.main()