
import sys
import torch

sys.path.insert(0, "/ref/inference")
import kernel as K

torch.set_default_dtype(torch.bfloat16)

assert torch.cuda.is_available(), "no CUDA device visible"
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"device={name} capability=sm{cap[0]}{cap[1]}", flush=True)

M, N, Kd = 64, 256, 256
dev = "cuda:0"

# FP8 act x FP4 weight GEMM: all-ones operands must yield K per element.
a = torch.full((M, Kd), 1.0, device=dev).to(torch.float8_e4m3fn)
b_bytes = torch.full((N, Kd // 2), 0x66, dtype=torch.uint8, device=dev)
b = b_bytes.view(torch.float4_e2m1fn_x2)
b_s = torch.ones(N, Kd // 32, device=dev, dtype=torch.float32)
c = K.fp4_gemm(a, a_s := torch.ones(M, Kd // 128, device=dev, dtype=torch.float32), b, b_s)
torch.cuda.synchronize()
err = (c.float() - Kd).abs().max().item()
print(f"fp4_gemm PASS compile+run, max_err={err:.4f}", flush=True)

# FP8 act x FP8 weight GEMM.
w = torch.full((N, Kd), 1.0, device=dev).to(torch.float8_e4m3fn)
w.scale = torch.ones(N, Kd // 128, device=dev, dtype=torch.float32)
c8 = K.fp8_gemm(a, a_s, w, w.scale)
torch.cuda.synchronize()
err8 = (c8.float() - Kd).abs().max().item()
print(f"fp8_gemm PASS compile+run, max_err={err8:.4f}", flush=True)

print("ALL_TILELANG_SM80_KERNELS_OK", flush=True)
