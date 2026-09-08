# CUDA Workspace Preallocation Benchmark Results

## Purpose

These local benchmarks compare the existing dynamic workspace allocation path with
run-scoped static workspace preallocation for CUDA `MatMulNBits`. The preallocated
path declares each kernel's workspace and includes its lifetime in ORT's activation
memory pattern.

The current RTX 5090 results use a generation workload with a 1,024-token
prefill followed by 128 chained batch-1 decode steps. It covers both fpA-intB
CUTLASS workspace and legacy dequantize-plus-cuBLAS workspace. The older T1000
prefill-only results are retained separately as historical data.

## Methodology

- Build: Release
- Runtime prepacking: enabled
- Device allocator for initializers: disabled
- CUDA arena extension strategy: `kSameAsRequested`
- Baseline and preallocated configurations: separate fresh processes

| Setting | RTX 5090 generation profile |
|---|---|
| Input | Batch 1, 1,024-token prefill followed by 128 deterministic decode tokens |
| Initial KV cache | Empty |
| Cache progression | Each decode step's present KV outputs become the next step's past KV inputs |
| Output | Full logits and present KV outputs for prefill and every decode step |
| Dispatch comparison | fpA-intB (`ep.cuda.fpa_intb_gemm=1`) and legacy (`=0`) |
| Warmup | 2 complete scenarios |
| Memory measurement | 1 complete scenario |
| Timed measurement | 10 complete scenarios |
| Reported latency | End-to-end scenario, prefill phase, and per-token decode distributions |

Each complete warmup traverses every decode-cache length. One test invocation
reports end-to-end, prefill, and decode timing for a generation scenario. Each
dispatch and preallocation configuration remains a separate fresh process so
arena regions and cached memory patterns from another configuration cannot
affect its measured high-water mark.

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

The initialization breakdown uses the CUDA allocator snapshot immediately after
`InferenceSession::Initialize()`:

- **Serialized external tensor data** is the size of the adjacent
  `model.onnx.data` file. It provides model-size context but is not a measurement
  of GPU-resident weight memory. Some models instead embed their weights in the
  `model.onnx` file, in which case the serialized ONNX size includes the weights.
- **Direct reserved bytes** are live allocations made through `IArena::Reserve()`.
  For these models they are primarily persistent `MatMulNBits` prepacked buffers,
  but the statistic is allocator-wide and is not exclusively a prepack counter.
- **BFC region capacity** is total allocated bytes minus direct reserved bytes.
- **Arena slack** is BFC region capacity minus live BFC bytes. It is unused
  capacity, not an external-fragmentation measurement.
- **Internal fragmentation** is `bytes_in_use - bytes_requested_in_use`; its
  ratio is that difference divided by `bytes_in_use`. It measures padding in live
  allocations. True external fragmentation would additionally require the
  largest free-chunk size, which allocator statistics do not currently expose.
- **Shrink reclaimed** is the reduction in total allocated bytes when
  `IArena::Shrink()` runs after warmup. `Shrink()` releases only completely free
  BFC regions; it does not release direct reserved allocations or compact live
  allocations.

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
- 141 nodes declared workspace in each generation profile

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| **Generation, fpA-intB path** | | | |
| Planned workspace nodes | 0 | 141 | +141 |
| Largest workspace | 0 B | 6,291,456 B (6.00 MiB) | +6,291,456 B |
| Post-initialization arena total | 2,162,425,088 B | 2,162,425,088 B | 0 B |
| Post-initialization direct reserved bytes | 818,712,576 B | 818,712,576 B | 0 B |
| Post-initialization BFC region capacity | 1,343,712,512 B | 1,343,712,512 B | 0 B |
| Post-initialization arena slack | 818,712,576 B | 818,712,576 B | 0 B |
| Shrink reclaimed after warmup | 1,129,877,504 B | 1,129,877,504 B | 0 B |
| Measured arena reservation | 420,757,760 B (401.27 MiB) | 421,855,744 B (402.31 MiB) | +1,097,984 B (+0.3%) |
| Final arena slack | 420,757,760 B | 421,855,744 B | +1,097,984 B |
| Internal fragmentation | 248 B | 248 B | 0 B |
| Internal fragmentation ratio | 0.0000185% | 0.0000185% | 0 pp |
| Arena allocation calls | 431,834 | 433,854 | +0.5% |
| WDDM initialization peak | 2,742 MiB | 2,742 MiB | 0 MiB |
| WDDM post-initialization usage | 2,644 MiB | 2,644 MiB | 0 MiB |
| WDDM pre-inference usage | 1,856 MiB | 1,856 MiB | 0 MiB |
| WDDM inference peak | 2,260 MiB | 2,270 MiB | +10 MiB |
| WDDM inference increase | 404 MiB | 414 MiB | +10 MiB |
| End-to-end average | 2,544.43 ms | 2,513.70 ms | -1.2% |
| End-to-end P50 | 2,434.93 ms | 2,424.55 ms | -0.4% |
| End-to-end P90 | 2,760.08 ms | 2,651.59 ms | -3.9% |
| Prefill average | 87.14 ms | 84.83 ms | -2.7% |
| Decode average per token | 19.20 ms | 18.98 ms | -1.2% |
| Decode P90 per token | 23.19 ms | 21.84 ms | -5.8% |
| Decode P99 per token | 30.20 ms | 29.26 ms | -3.1% |
| Initialization | 57.87 s | 67.75 s | +17.1% |
| **Generation, legacy path** | | | |
| Planned workspace nodes | 0 | 141 | +141 |
| Largest workspace | 0 B | 100,663,296 B (96.00 MiB) | +100,663,296 B |
| Post-initialization arena total | 1,343,712,512 B | 1,343,712,512 B | 0 B |
| Post-initialization direct reserved bytes | 0 B | 0 B | 0 B |
| Post-initialization BFC region capacity | 1,343,712,512 B | 1,343,712,512 B | 0 B |
| Post-initialization arena slack | 0 B | 0 B | 0 B |
| Shrink reclaimed after warmup | 689,979,392 B | 900,481,024 B | +210,501,632 B |
| Measured arena reservation | 545,014,016 B (519.77 MiB) | 450,691,328 B (429.81 MiB) | **-94,322,688 B (-17.3%)** |
| Final arena slack | 545,014,016 B | 450,691,328 B | **-94,322,688 B** |
| Internal fragmentation | 248 B | 248 B | 0 B |
| Internal fragmentation ratio | 0.0000185% | 0.0000185% | 0 pp |
| Arena allocation calls | 431,378 | 429,686 | -0.4% |
| WDDM initialization peak | 1,706 MiB | 1,706 MiB | 0 MiB |
| WDDM post-initialization usage | 1,706 MiB | 1,706 MiB | 0 MiB |
| WDDM pre-inference usage | 1,772 MiB | 1,772 MiB | 0 MiB |
| WDDM inference peak | 2,298 MiB | 2,212 MiB | **-86 MiB** |
| WDDM inference increase | 526 MiB | 440 MiB | **-86 MiB** |
| End-to-end average | 2,465.07 ms | 2,453.29 ms | -0.5% |
| End-to-end P50 | 2,407.36 ms | 2,449.66 ms | +1.8% |
| End-to-end P90 | 2,568.17 ms | 2,526.84 ms | -1.6% |
| Prefill average | 87.09 ms | 88.27 ms | +1.4% |
| Decode average per token | 18.58 ms | 18.48 ms | -0.5% |
| Decode P90 per token | 21.11 ms | 20.93 ms | -0.8% |
| Decode P99 per token | 28.88 ms | 30.88 ms | +6.9% |
| Initialization | 1.80 s | 1.68 s | -6.7% |

fpA-intB preallocation did not reduce the memory high-water mark. Legacy
preallocation reduced measured arena reservation by
94,322,688 bytes (89.95 MiB) and WDDM inference peak by 86 MiB, close to its
96 MiB largest declared workspace. The generation latency changes were small and
mixed, so the legacy memory reduction is the primary result.

### Qwen 2.5 7B

Model:
`qwen2.5-7b-instruct-cuda-gpu:4`

- 28 decoder layers
- 4 key/value heads
- 141 CUDA `MatMulNBits` nodes
- 141 nodes declared workspace in each generation profile

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| **Generation, fpA-intB path** | | | |
| Planned workspace nodes | 0 | 141 | +141 |
| Largest workspace | 0 B | 33,030,144 B (31.50 MiB) | +33,030,144 B |
| Post-initialization arena total | 8,793,038,080 B | 8,793,038,080 B | 0 B |
| Post-initialization direct reserved bytes | 3,716,923,392 B | 3,716,923,392 B | 0 B |
| Post-initialization BFC region capacity | 5,076,114,688 B | 5,076,114,688 B | 0 B |
| Post-initialization arena slack | 3,716,923,392 B | 3,716,923,392 B | 0 B |
| Shrink reclaimed after warmup | 4,028,350,464 B | 4,028,350,464 B | 0 B |
| Measured arena reservation | 565,723,392 B (539.52 MiB) | 567,935,488 B (541.63 MiB) | +2,212,096 B (+0.4%) |
| Final arena slack | 565,723,392 B | 567,935,488 B | +2,212,096 B |
| Internal fragmentation | 248 B | 248 B | 0 B |
| Internal fragmentation ratio | 0.00000489% | 0.00000489% | 0 pp |
| Arena allocation calls | 433,498 | 430,270 | -0.7% |
| WDDM initialization peak | 9,650 MiB | 9,650 MiB | 0 MiB |
| WDDM post-initialization usage | 9,424 MiB | 9,424 MiB | 0 MiB |
| WDDM pre-inference usage | 5,598 MiB | 5,598 MiB | 0 MiB |
| WDDM inference peak | 6,142 MiB | 6,148 MiB | +6 MiB |
| WDDM inference increase | 544 MiB | 550 MiB | +6 MiB |
| End-to-end average | 4,428.60 ms | 4,466.76 ms | +0.9% |
| End-to-end P50 | 4,421.69 ms | 4,445.41 ms | +0.5% |
| End-to-end P90 | 4,458.47 ms | 4,506.49 ms | +1.1% |
| Prefill average | 250.44 ms | 251.37 ms | +0.4% |
| Decode average per token | 32.64 ms | 32.93 ms | +0.9% |
| Decode P90 per token | 34.49 ms | 35.01 ms | +1.5% |
| Decode P99 per token | 37.33 ms | 37.56 ms | +0.6% |
| Initialization | 107.39 s | 116.42 s | +8.4% |
| **Generation, legacy path** | | | |
| Planned workspace nodes | 0 | 141 | +141 |
| Largest workspace | 0 B | 234,881,024 B (224.00 MiB) | +234,881,024 B |
| Post-initialization arena total | 5,076,114,688 B | 5,076,114,688 B | 0 B |
| Post-initialization direct reserved bytes | 0 B | 0 B | 0 B |
| Post-initialization BFC region capacity | 5,076,114,688 B | 5,076,114,688 B | 0 B |
| Post-initialization arena slack | 0 B | 0 B | 0 B |
| Shrink reclaimed after warmup | 1,174,421,504 B | 1,250,967,552 B | +76,546,048 B |
| Measured arena reservation | 1,006,125,312 B (959.52 MiB) | 670,695,936 B (639.63 MiB) | **-335,429,376 B (-33.3%)** |
| Final arena slack | 1,006,125,312 B | 670,695,936 B | **-335,429,376 B** |
| Internal fragmentation | 248 B | 248 B | 0 B |
| Internal fragmentation ratio | 0.00000489% | 0.00000489% | 0 pp |
| Arena allocation calls | 431,378 | 429,686 | -0.4% |
| WDDM initialization peak | 5,490 MiB | 5,490 MiB | 0 MiB |
| WDDM post-initialization usage | 5,490 MiB | 5,490 MiB | 0 MiB |
| WDDM pre-inference usage | 5,556 MiB | 5,556 MiB | 0 MiB |
| WDDM inference peak | 6,520 MiB | 6,206 MiB | **-314 MiB** |
| WDDM inference increase | 964 MiB | 650 MiB | **-314 MiB** |
| End-to-end average | 4,620.63 ms | 4,555.02 ms | -1.4% |
| End-to-end P50 | 4,604.12 ms | 4,548.14 ms | -1.2% |
| End-to-end P90 | 4,683.44 ms | 4,606.32 ms | -1.6% |
| Prefill average | 276.13 ms | 274.15 ms | -0.7% |
| Decode average per token | 33.94 ms | 33.44 ms | -1.5% |
| Decode P90 per token | 36.66 ms | 35.78 ms | -2.4% |
| Decode P99 per token | 41.13 ms | 39.92 ms | -3.0% |
| Initialization | 4.16 s | 3.75 s | -9.8% |

fpA-intB preallocation did not reduce memory. Legacy preallocation reduced
measured arena reservation by 335,429,376 bytes
(319.89 MiB) and WDDM inference peak by 314 MiB. The arena change exceeds the
224 MiB largest individual workspace because memory-pattern placement also
changed BFC region packing and the allocation high-water mark. The generation
latency differences remain single-pair observations; the legacy memory reduction
is the primary result.

### RTX 5090 summary

| Model | Workload and path | WDDM inference-peak change | Arena reservation change | Allocation-call change | Average-latency change |
|---|---|---:|---:|---:|---:|
| Qwen 2.5 1.5B | Generation, fpA-intB | +10 MiB | +1,097,984 B | +0.5% | -1.2% |
| Qwen 2.5 1.5B | Generation, legacy | **-86 MiB** | **-94,322,688 B** | -0.4% | -0.5% |
| Qwen 2.5 7B | Generation, fpA-intB | +6 MiB | +2,212,096 B | -0.7% | +0.9% |
| Qwen 2.5 7B | Generation, legacy | **-314 MiB** | **-335,429,376 B** | -0.4% | -1.4% |

The generation workloads show the strongest memory benefit when fpA-intB is
disabled and preallocation covers the legacy dequantized-weight workspace. The
legacy path reduced WDDM inference peak by 86 MiB for Qwen 2.5 1.5B and 314 MiB
for Qwen 2.5 7B. fpA-intB generation did not reduce the memory high-water mark
for either model because the growing KV cache and workspace-free decode steps
dominated the complete scenario. Latency moved in different directions across
the single paired runs.

## Historical NVIDIA T1000 prefill-only results

The T1000 runs predate both the initialization-breakdown instrumentation and
legacy workspace declaration. Their serialized-size, direct-reserve,
arena-slack, and fragmentation metrics were not recorded, so the original
results are preserved below without inferred values and must not be compared to
the current RTX 5090 planned-node counts.

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

## CUDA MatMulNBits implementation count

The CUDA `MatMulNBits::ComputeInternal()` dispatcher has five conceptual runtime
implementation families:

| Implementation | Per-run temporary buffers | Benefits from workspace preallocation |
|---|---|---|
| fpA-intB CUDA GEMV | None | No current benefit |
| fpA-intB CUTLASS GEMM | CUTLASS runner workspace | **Yes** |
| Fused small-M CUDA kernels | None allocated from the device allocator | No current benefit |
| Full dequantize plus cuBLAS GEMM | Full dequantized weight matrix | **Yes** |
| Chunked dequantize plus cuBLAS GEMM | One dequantized weight chunk | **Yes** |

Therefore, three of the five runtime implementation families currently consume
planned workspace: fpA-intB CUTLASS GEMM, full dequantize plus cuBLAS GEMM, and
chunked dequantize plus cuBLAS GEMM. The two GEMV families remain workspace-free.

The fpA-intB path profiles tactics and selects between two execution families:

- The CUDA GEMV tactic is intended for small `M`, including ordinary batch-1
  decode. It consumes the persistent prepacked weights, scales, and optional
  zero points directly and does not request a run-scoped workspace.
- The CUTLASS GEMM tactic is normally selected for larger `M`, including the
  prefill workloads measured above. Its runner reports a workspace size for the
  selected dimensions and tactic. This is the workspace covered by the current
  benchmark and preallocation implementation.

CUTLASS has SM80-compatible and native SM90 kernel/configuration variants, but
they are one conceptual implementation family for workspace planning. Both use
the same runner workspace interface, although the required size depends on the
effective architecture and selected tactic.

The fused small-M family is reached when the fpA-intB path is unavailable. It
contains several specialized kernels, including 4-bit router GEMV, 4-bit and
8-bit single-row kernels, and batched/small-M variants. These use registers or
CUDA shared memory rather than allocator-backed temporary buffers, so planned
workspace would not remove a device allocation.

If neither optimized family applies, CUDA dequantizes the quantized weight into
a temporary floating-point buffer and invokes cuBLAS GEMM. The normal fallback
materializes the full `[N, K_padded]` matrix. The chunked fallback reduces peak
scratch by dequantizing and multiplying one range of `N` rows at a time. Both
declare the required scratch size and first request their buffer from the
preallocated workspace. When no planned workspace is available, they fall back
to a dynamic scratch allocation.

Initialization-time weight prepacking is separate from these runtime
implementations. Its persistent packed weights and temporary conversion or
profiling buffers are created during session initialization and are not
run-scoped workspace covered by this benchmark.
