import torch
import time

size = 3000
device_gpu = "mps"

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
_ = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.backends.mps.is_available():
    a_gpu = a_cpu.to(device_gpu)
    b_gpu = b_cpu.to(device_gpu)

    start = time.time()
    _ = a_gpu @ b_gpu
    gpu_time = time.time() - start

    print(f"MPS GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
else:
    print("MPS not available")
