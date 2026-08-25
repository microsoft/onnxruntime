# CUDA Workspace Preallocation Benchmark Results

## Purpose

These local benchmarks compare the existing dynamic workspace allocation path with
run-scoped static workspace preallocation for CUDA `MatMulNBits`. The preallocated
path declares each kernel's workspace and includes its lifetime in ORT's activation
memory pattern.

## Methodology

- Build: Release
- GPU: NVIDIA T1000, SM75, 4 GiB
- CUDA toolkit: 12.8
- Input: batch 1, sequence length 64, past sequence length 1
- `ep.cuda.fpa_intb_gemm`: enabled
- Runtime prepacking: enabled
- Device allocator for initializers: disabled
- CUDA arena extension strategy: `kSameAsRequested`
- Warmup runs: 5
- Memory-measurement runs: 3
- Timed runs: 30
- Baseline and preallocated configurations ran in separate fresh processes.

The **measured arena reservation** is the increase in
`AllocatorStats::total_allocated_bytes` during the post-warmup measurement runs.
Before measuring, the test calls `IArena::Shrink()` to release free first-run
regions. With `kSameAsRequested`, this exposes the CUDA reservation attributable
to the cached activation pattern and any separate workspace allocation.

`cudaMemGetInfo` is also sampled, but its process-wide peak is too coarse and noisy
to resolve workspace-sized differences. The arena reservation is the primary
memory comparison.

## Qwen 2.5 1.5B

Model:
`qwen2.5-1.5b-instruct-cuda-gpu:4`

- 28 decoder layers
- 2 key/value heads
- 141 CUDA `MatMulNBits` nodes
- 113 nodes declared nonzero workspace for the tested shape

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| Planned workspace nodes | 0 | 113 | +113 |
| Largest workspace | 0 B | 265,984 B | +265,984 B |
| Measured arena reservation | 27,062,016 B | 26,811,904 B | **-250,112 B** |
| Arena allocation calls | 15,000 | 10,819 | **-27.9%** |
| Inference peak GPU usage | 2,310.94 MiB | 2,308.94 MiB | -2.00 MiB |
| Average latency | 350.22 ms | 356.39 ms | +1.76% |
| P50 latency | 350.55 ms | 356.52 ms | +1.70% |
| P90 latency | 351.35 ms | 360.48 ms | +2.60% |
| Initialization | 25.09 min | 25.23 min | +0.56% |

Workspace preallocation reduced the controlled CUDA arena reservation by
250,112 bytes (approximately 244 KiB). This is close to the largest declared
workspace of 265,984 bytes, indicating that most workspace storage overlapped
non-live activation memory. It also eliminated 4,181 allocator calls. This run
showed a small latency regression.

## Hy-MT2 1.8B

Model:
`Hy-MT2-1.8B-ONNX/Q4_KQuant_tie/cuda`

- 32 decoder layers
- 4 key/value heads
- 225 CUDA `MatMulNBits` nodes
- 224 nodes declared nonzero workspace for the tested shape

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| Planned workspace nodes | 0 | 224 | +224 |
| Largest workspace | 0 B | 10,752 B | +10,752 B |
| Measured arena reservation | 159,790,336 B | 159,797,760 B | **+7,424 B** |
| Arena allocation calls | 20,407 | 12,119 | **-40.6%** |
| Inference peak GPU usage | 2,244.94 MiB | 2,244.94 MiB | 0 MiB |
| Average latency | 470.15 ms | 466.97 ms | -0.68% |
| P50 latency | 469.63 ms | 466.59 ms | -0.65% |
| P90 latency | 475.40 ms | 468.67 ms | -1.42% |
| Initialization | 2.31 min | 2.30 min | -0.50% |

Workspace preallocation did not reduce the controlled CUDA arena reservation for
this shape. It added 7,424 bytes, while eliminating 8,288 allocator calls. The
Release run showed a small latency improvement.

## Summary

| Model | Arena reservation change | Allocation-call change | Average-latency change |
|---|---:|---:|---:|
| Qwen 2.5 1.5B | **-250,112 B** | **-27.9%** | +1.76% |
| Hy-MT2 1.8B | +7,424 B | **-40.6%** | -0.68% |

The memory benefit is model- and shape-dependent. Qwen exposes enough workspace
to reuse approximately 244 KiB of activation storage, while Hy-MT2's much smaller
workspace does not produce a net reservation reduction at the tested shape. Both
models substantially reduce allocator calls.
