# MatMulNBits Small-M GEMV Profiling Experiments

This file records CUDA `MatMulNBits` 4-bit profiling results for the small-M
range (M = 2..16, e.g. multi-row decode or short prefill) so future
kernel and dispatch changes can be compared against a stable baseline.

> **Note**: These are **point-in-time** measurements captured on the specific GPU, driver, CUDA
> toolkit, and ORT build noted in the section header. Treat the numbers as a historical baseline
> for regression comparison, not as current performance guidance — re-run the benchmark script on
> your own hardware before drawing conclusions.

## 2026-06-30 Baseline: A100 (SM80), Warmup 25, Repeat 200

### Setup

- Machine/GPU: A100 (SM80).
- CUDA toolkit: 13.0. ONNX Runtime build: `build/Release`.
- Benchmark script: `onnxruntime/test/python/transformers/profile_matmul_nbits.py`.
- Warmup: 25 ORT runs before the measured `benchmark` NVTX range. Repeat: best of 10 trials x 200 runs.
- Weights: 4-bit, block_size 32, asymmetric (with zero points), per the standard `[N, blocks, blob]` layout.
- Matrices sized after a Qwen3-8B-class dense decoder + lm_head.

Command template:

```bash
# Host-timing table across all matrices:
python onnxruntime/test/python/transformers/profile_matmul_nbits.py --warmup 25 --repeat 200

# Single case:
python onnxruntime/test/python/transformers/profile_matmul_nbits.py --k 4096 --n 12288 --m 8

# Kernel-level via nsys + the repo parser:
nsys profile -t cuda,nvtx -o mnb --export=sqlite \
    python onnxruntime/test/python/transformers/profile_matmul_nbits.py --k 4096 --n 12288 --m 8
python onnxruntime/test/python/transformers/parse_nsys.py mnb.sqlite --nvtx-range benchmark --pattern '%'
```

### Before / After

`before` is the prior dispatch (M=1 single-row GEMV; M>1 falls back to weight dequantization + cuBLAS
GEMM, which is M-independent and dequantizes the full weight regardless of M). `after` is the batched
small-M GEMV (`MatMulFloat4BatchedKernel`) covering M=2..16. Values are average op latency in microseconds
(lower is better).

Before (dequant + cuBLAS for M>1):

| matrix  | K     | N      | M=1   | M=2    | M=4    | M=8    | M=16   |
|---------|-------|--------|-------|--------|--------|--------|--------|
| qkv     | 4096  | 4096   | 25.9  | 77.2   | 69.3   | 69.5   | 69.9   |
| o_proj  | 4096  | 4096   | 23.2  | 69.2   | 69.1   | 69.3   | 69.7   |
| gate_up | 4096  | 12288  | 39.3  | 172.0  | 172.1  | 172.1  | 172.4  |
| down    | 12288 | 4096   | 38.8  | 174.8  | 175.1  | 175.4  | 175.6  |
| lm_head | 4096  | 151936 | 301.6 | 1868.0 | 1871.1 | 1877.8 | 1885.1 |

After (batched small-M GEMV for M=2..16):

| matrix  | K     | N      | M=1   | M=2   | M=4   | M=8   | M=16   |
|---------|-------|--------|-------|-------|-------|-------|--------|
| qkv     | 4096  | 4096   | 25.4  | 30.1  | 32.3  | 43.2  | 70.0   |
| o_proj  | 4096  | 4096   | 22.6  | 26.4  | 28.1  | 36.7  | 58.7   |
| gate_up | 4096  | 12288  | 37.5  | 43.5  | 49.5  | 71.9  | 125.9  |
| down    | 12288 | 4096   | 37.9  | 47.4  | 52.9  | 76.8  | 150.1  |
| lm_head | 4096  | 151936 | 300.7 | 329.9 | 424.1 | 635.3 | 1226.9 |

Speedup (before / after, >1 means the batched GEMV is faster):

| matrix  | M=2   | M=4   | M=8   | M=16  |
|---------|-------|-------|-------|-------|
| qkv     | 2.56x | 2.15x | 1.61x | 1.00x |
| o_proj  | 2.62x | 2.46x | 1.89x | 1.19x |
| gate_up | 3.95x | 3.48x | 2.39x | 1.37x |
| down    | 3.69x | 3.31x | 2.28x | 1.17x |
| lm_head | 5.66x | 4.41x | 2.96x | 1.54x |

### Observations

- The previous M>1 path dequantizes the entire weight matrix to a temporary buffer and calls cuBLAS, so
  its latency is flat across M (e.g. `gate_up` ~172 us, `lm_head` ~1.9 ms even at M=2). The batched small-M
  GEMV reads the quantized weight once and scales with M, giving 2.6-5.7x at M=2 and staying at or above
  parity through M=16.
- M=1 decode is unchanged (same single-row GEMV in both builds).
- No prepacking is used, so there is no extra resident weight memory and no GEMM tactic profiling at
  session init.

### Next Experiments

- Sweep block_size {16, 32, 64, 128} and bf16 activations.
- Add kernel-level (nsys) breakdowns to separate compute from launch/dispatch overhead.
