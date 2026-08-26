# CUDA Workspace Preallocation Benchmark Results

## Purpose

These local benchmarks compare the existing dynamic workspace allocation path with
run-scoped static workspace preallocation for CUDA `MatMulNBits`. The preallocated
path declares each kernel's workspace and includes its lifetime in ORT's activation
memory pattern.

## Methodology

- Build: Release
- Input: batch 1, sequence length 64, past sequence length 1
- `ep.cuda.fpa_intb_gemm`: enabled
- Runtime prepacking: enabled
- Device allocator for initializers: disabled
- CUDA arena extension strategy: `kSameAsRequested`
- Warmup runs: 5
- Memory-measurement runs: 3
- Timed runs: 30
- Baseline and preallocated configurations ran in separate fresh processes.

| GPU | Compute capability | CUDA toolkit | Driver |
|---|---:|---:|---:|
| NVIDIA GeForce RTX 5090 Laptop GPU | SM120 | 13.3 | 610.62 |
| NVIDIA T1000, 4 GiB | SM75 | 12.8 | Not recorded |

The **WDDM process peak** is sampled every 5 ms from
`IDXGIAdapter3::QueryVideoMemoryInfo(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)`. The DXGI
adapter is matched to CUDA device 0 by LUID. `CurrentUsage` measures local video
memory attributed to the benchmark process, excluding desktop composition and
unrelated GPU processes.

The **measured arena reservation** is the increase in
`AllocatorStats::total_allocated_bytes` during the post-warmup measurement runs.
Before measuring, the test calls `IArena::Shrink()` to release free first-run
regions. With `kSameAsRequested`, this exposes the CUDA reservation attributable
to the cached activation pattern and any separate workspace allocation.

`cudaMemGetInfo` is also sampled, but it reports device-wide usage and includes
unrelated processes. WDDM is the primary process-peak measurement; the arena
reservation isolates the exact effect inside ORT. A lower arena reservation does
not necessarily lower WDDM peak VRAM if another point in execution remains the
high-water mark or if the difference is below WDDM accounting granularity.

## NVIDIA GeForce RTX 5090 Laptop GPU

### Qwen 2.5 1.5B

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
| Measured arena reservation | 26,662,912 B | 26,412,800 B | **-250,112 B** |
| Arena allocation calls | 13,936 | 9,755 | **-30.0%** |
| WDDM initialization peak | 2,798 MiB | 2,798 MiB | **0 MiB** |
| WDDM post-initialization usage | 2,700 MiB | 2,700 MiB | **0 MiB** |
| WDDM pre-inference usage | 1,834 MiB | 1,834 MiB | **0 MiB** |
| WDDM inference peak | 1,860 MiB | 1,860 MiB | **0 MiB** |
| WDDM inference increase | 26 MiB | 26 MiB | **0 MiB** |
| Average latency | 27.63 ms | 20.77 ms | **-24.8%** |
| P50 latency | 23.67 ms | 13.50 ms | **-43.0%** |
| P90 latency | 39.37 ms | 30.56 ms | **-22.4%** |
| Initialization | 90.48 s | 68.84 s | **-23.9%** |

Workspace preallocation reduced the controlled CUDA arena reservation by
250,112 bytes (approximately 244 KiB). This is close to the largest declared
workspace of 265,984 bytes, indicating that most workspace storage overlapped
non-live activation memory. It also eliminated 4,181 allocator calls. The WDDM
process peak did not change: the 244 KiB arena saving is below its reported MiB
granularity and did not move the workload's overall high-water mark. In this run,
average latency decreased by 24.8%.

### Hy-MT2 1.8B

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
| Measured arena reservation | 159,454,464 B | 159,462,144 B | **+7,680 B** |
| Arena allocation calls | 19,191 | 10,903 | **-43.2%** |
| WDDM initialization peak | 2,406 MiB | 2,406 MiB | **0 MiB** |
| WDDM post-initialization usage | 2,406 MiB | 2,406 MiB | **0 MiB** |
| WDDM pre-inference usage | 1,658 MiB | 1,658 MiB | **0 MiB** |
| WDDM inference peak | 1,806 MiB | 1,806 MiB | **0 MiB** |
| WDDM inference increase | 148 MiB | 148 MiB | **0 MiB** |
| Average latency | 17.91 ms | 15.33 ms | **-14.4%** |
| P50 latency | 17.48 ms | 14.70 ms | **-15.9%** |
| P90 latency | 20.66 ms | 16.36 ms | **-20.8%** |
| Initialization | 9.01 s | 6.64 s | **-26.3%** |

Workspace preallocation did not reduce the controlled CUDA arena reservation for
this shape. It added 7,680 bytes, while eliminating 8,288 allocator calls. The
WDDM process peaks were identical. In this run, average latency decreased by
14.4%.

### RTX 5090 summary

| Model | WDDM inference-peak change | Arena reservation change | Allocation-call change | Average-latency change |
|---|---:|---:|---:|---:|
| Qwen 2.5 1.5B | **0 MiB** | **-250,112 B** | **-30.0%** | **-24.8%** |
| Hy-MT2 1.8B | **0 MiB** | +7,680 B | **-43.2%** | **-14.4%** |

The memory benefit is model- and shape-dependent. Qwen exposes enough workspace
to reuse approximately 244 KiB of activation storage, while Hy-MT2's much smaller
workspace does not produce a net reservation reduction at the tested shape. Both
models substantially reduce allocator calls. Neither change moved the
process-scoped WDDM peak for this workload and GPU. Both paired runs measured
lower latency with workspace preallocation on the RTX 5090 Laptop GPU.

## NVIDIA T1000

### Qwen 2.5 1.5B

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
| WDDM initialization peak | 2,546 MiB | 2,546 MiB | **0 MiB** |
| WDDM post-initialization usage | 2,448 MiB | 2,448 MiB | **0 MiB** |
| WDDM pre-inference usage | 1,532 MiB | 1,532 MiB | **0 MiB** |
| WDDM inference peak | 1,558 MiB | 1,558 MiB | **0 MiB** |
| WDDM inference increase | 26 MiB | 26 MiB | **0 MiB** |
| Average latency | 349.99 ms | 349.41 ms | -0.16% |
| P50 latency | 349.98 ms | 349.80 ms | -0.05% |
| P90 latency | 351.91 ms | 350.67 ms | -0.35% |
| Initialization | 25.07 min | 25.13 min | +0.23% |

Workspace preallocation reduced the controlled CUDA arena reservation by
250,112 bytes (approximately 244 KiB). This is close to the largest declared
workspace of 265,984 bytes, indicating that most workspace storage overlapped
non-live activation memory. It also eliminated 4,181 allocator calls. The WDDM
process peak did not change: the 244 KiB arena saving is below its reported MiB
granularity and did not move the workload's overall high-water mark. The paired
latency measurements were effectively unchanged.

### Hy-MT2 1.8B

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
| WDDM initialization peak | 2,154 MiB | 2,154 MiB | **0 MiB** |
| WDDM post-initialization usage | 2,154 MiB | 2,154 MiB | **0 MiB** |
| WDDM pre-inference usage | 1,348 MiB | 1,348 MiB | **0 MiB** |
| WDDM inference peak | 1,496 MiB | 1,496 MiB | **0 MiB** |
| WDDM inference increase | 148 MiB | 148 MiB | **0 MiB** |
| Average latency | 466.04 ms | 466.79 ms | +0.16% |
| P50 latency | 465.96 ms | 466.61 ms | +0.14% |
| P90 latency | 468.10 ms | 469.15 ms | +0.22% |
| Initialization | 2.34 min | 2.29 min | -2.13% |

Workspace preallocation did not reduce the controlled CUDA arena reservation for
this shape. It added 7,424 bytes, while eliminating 8,288 allocator calls. The
WDDM process peaks were identical, and paired latency was effectively unchanged.

### T1000 summary

| Model | WDDM inference-peak change | Arena reservation change | Allocation-call change | Average-latency change |
|---|---:|---:|---:|---:|
| Qwen 2.5 1.5B | **0 MiB** | **-250,112 B** | **-27.9%** | -0.16% |
| Hy-MT2 1.8B | **0 MiB** | +7,424 B | **-40.6%** | +0.16% |

On the T1000, Qwen reused approximately 244 KiB of activation storage, while
Hy-MT2's smaller workspace did not produce a net reservation reduction. Both
models substantially reduced allocator calls without moving the process-scoped
WDDM peak, and latency was effectively unchanged.
