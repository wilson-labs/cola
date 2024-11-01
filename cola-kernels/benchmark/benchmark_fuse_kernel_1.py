import torch
import torch.profiler as profiler
import cola_kernels

n = 1024
A = torch.randn((n, n), device='cuda')
B = torch.randn((n, n), device='cuda')
C = torch.randn((n, n), device='cuda')

with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
    D = A * B + C

print(prof.key_averages().table(sort_by="cuda_time_total"))

with profiler.profile(activities=[profiler.ProfilerActivity.CUDA]) as prof:
    D = cola_kernels.ops.fuse_kernel_1(A, B, C)

print(prof.key_averages().table(sort_by="cuda_time_total"))