# CUDA Workspace Preallocation Benchmark Results

## Purpose

These local benchmarks compare the existing dynamic workspace allocation path with
run-scoped static workspace preallocation for CUDA `MatMulNBits`. The preallocated
path declares each kernel's workspace and includes its lifetime in ORT's activation
memory pattern.

These results cover **prefill only**, not token-by-token decode. Each run processes
a prefill chunk with a one-token KV cache, which selects the fpA-intB GEMM path
whose CUTLASS workspace is targeted by this feature. The RTX 5090 runs use 1,024
new tokens to represent a long-prompt prefill workload. The preserved T1000 runs
used 64 new tokens. Ordinary batch-1 decode processes one new token and typically
selects the workspace-free GEMV path, so the latency changes reported here should
not be interpreted as decode improvements.

## Methodology

- Build: Release
- Workload: synthetic cached prefill
- RTX 5090 input IDs: batch 1, 1,024 new tokens, all set to the model's BOS token
- RTX 5090 attention mask: shape `[1, 1025]`, all valid tokens
- T1000 input IDs: batch 1, 64 new tokens, all set to the model's BOS token
- T1000 attention mask: shape `[1, 65]`, all valid tokens
- Past key/value cache: one zero-filled token per layer
- Output: logits for all new tokens
- `ep.cuda.fpa_intb_gemm`: enabled
- Runtime prepacking: enabled
- Device allocator for initializers: disabled
- CUDA arena extension strategy: `kSameAsRequested`
- Warmup runs: 5
- Memory-measurement runs: 3
- Timed runs: 30
- Baseline and preallocated configurations ran in separate fresh processes.

Warmup, memory-measurement, and timed iterations all reuse the same feeds. Present
key/value outputs are not fed into the next iteration, so the benchmark does not
simulate a growing KV cache or an autoregressive generation loop.

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
- 113 nodes declared nonzero workspace for the tested shape

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| Planned workspace nodes | 0 | 113 | +113 |
| Largest workspace | 0 B | 4,254,208 B | +4,254,208 B |
| Serialized ONNX graph | 197,130 B | 197,130 B | 0 B |
| Serialized external tensor data | 1,343,683,584 B (1,281.44 MiB) | 1,343,683,584 B (1,281.44 MiB) | 0 B |
| Post-initialization arena total | 2,211,970,304 B (2,109.50 MiB) | 2,211,970,304 B (2,109.50 MiB) | 0 B |
| Post-initialization direct reserved bytes | 868,257,792 B (828.04 MiB) | 868,257,792 B (828.04 MiB) | 0 B |
| Post-initialization BFC region capacity | 1,343,712,512 B (1,281.46 MiB) | 1,343,712,512 B (1,281.46 MiB) | 0 B |
| Post-initialization arena slack | 868,257,792 B (828.04 MiB) | 868,257,792 B (828.04 MiB) | 0 B |
| Shrink reclaimed after warmup | 1,179,422,720 B (1,124.79 MiB) | 1,179,422,720 B (1,124.79 MiB) | 0 B |
| Measured arena reservation | 426,148,096 B | 422,144,768 B | **-4,003,328 B** |
| Final arena slack | 426,148,096 B | 422,144,768 B | **-4,003,328 B** |
| Internal fragmentation | 248 B | 248 B | 0 B |
| Internal fragmentation ratio | 0.0000185% | 0.0000185% | 0 pp |
| Arena allocation calls | 13,936 | 9,755 | **-30.0%** |
| WDDM initialization peak | 2,798 MiB | 2,798 MiB | **0 MiB** |
| WDDM post-initialization usage | 2,700 MiB | 2,700 MiB | **0 MiB** |
| WDDM pre-inference usage | 1,838 MiB | 1,838 MiB | **0 MiB** |
| WDDM inference peak | 2,260 MiB | 2,254 MiB | **-6 MiB** |
| WDDM inference increase | 422 MiB | 416 MiB | **-6 MiB** |
| Average latency | 110.82 ms | 92.48 ms | **-16.5%** |
| P50 latency | 108.49 ms | 91.29 ms | **-15.8%** |
| P90 latency | 121.77 ms | 97.79 ms | **-19.7%** |
| Initialization | 57.70 s | 76.86 s | +33.2% |

Workspace preallocation reduced the controlled CUDA arena reservation by
4,003,328 bytes (approximately 3.82 MiB). This is close to the largest declared
workspace of 4,254,208 bytes (approximately 4.06 MiB), indicating that most
workspace storage overlapped non-live activation memory. It also eliminated
4,181 allocator calls and reduced the WDDM inference peak by 6 MiB.

The 1,281.44 MiB serialized external-data file closely matches the 1,281.46 MiB
BFC region capacity created during initialization. After prepacking and
initialization cleanup, 828.04 MiB of that capacity was unused and was later
released by `Shrink()`. The separate 828.04 MiB of live direct reserves remained
allocated; these are primarily persistent prepacked buffers. Internal
fragmentation was only 248 bytes, so the large post-initialization slack reflects
free BFC capacity rather than live-allocation padding. The WDDM process peak did
not change between configurations.

### Qwen 2.5 7B

Model:
`qwen2.5-7b-instruct-cuda-gpu:4`

- 28 decoder layers
- 4 key/value heads
- 141 CUDA `MatMulNBits` nodes
- 113 nodes declared nonzero workspace for the tested shape

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| Planned workspace nodes | 0 | 113 | +113 |
| Largest workspace | 0 B | 4,257,792 B | +4,257,792 B |
| Serialized ONNX graph | 198,083 B | 198,083 B | 0 B |
| Serialized external tensor data | 5,076,085,760 B (4,840.93 MiB) | 5,076,085,760 B (4,840.93 MiB) | 0 B |
| Post-initialization arena total | 9,053,150,464 B (8,633.76 MiB) | 9,053,150,464 B (8,633.76 MiB) | 0 B |
| Post-initialization direct reserved bytes | 3,977,035,776 B (3,792.80 MiB) | 3,977,035,776 B (3,792.80 MiB) | 0 B |
| Post-initialization BFC region capacity | 5,076,114,688 B (4,840.96 MiB) | 5,076,114,688 B (4,840.96 MiB) | 0 B |
| Post-initialization arena slack | 3,977,035,776 B (3,792.80 MiB) | 3,977,035,776 B (3,792.80 MiB) | 0 B |
| Shrink reclaimed after warmup | 4,288,462,848 B (4,089.80 MiB) | 4,288,462,848 B (4,089.80 MiB) | 0 B |
| Measured arena reservation | 542,319,360 B | 542,849,792 B | **+530,432 B** |
| Final arena slack | 542,319,360 B | 542,849,792 B | **+530,432 B** |
| Internal fragmentation | 248 B | 248 B | 0 B |
| Internal fragmentation ratio | 0.00000489% | 0.00000489% | 0 pp |
| Arena allocation calls | 13,936 | 9,755 | **-30.0%** |
| WDDM initialization peak | 9,902 MiB | 9,902 MiB | **0 MiB** |
| WDDM post-initialization usage | 9,676 MiB | 9,676 MiB | **0 MiB** |
| WDDM pre-inference usage | 5,604 MiB | 5,604 MiB | **0 MiB** |
| WDDM inference peak | 6,132 MiB | 6,132 MiB | **0 MiB** |
| WDDM inference increase | 528 MiB | 528 MiB | **0 MiB** |
| Average latency | 261.47 ms | 259.04 ms | -0.9% |
| P50 latency | 260.47 ms | 258.10 ms | -0.9% |
| P90 latency | 264.95 ms | 267.16 ms | +0.8% |
| Initialization | 111.91 s | 142.11 s | +27.0% |

Workspace preallocation did not reduce the controlled CUDA arena reservation for
this model and shape. It added 530,432 bytes (approximately 518 KiB), despite a
largest declared workspace of approximately 4.06 MiB, while eliminating 4,181
allocator calls. The process-scoped WDDM inference peak was unchanged, and the
paired latency measurements were effectively unchanged.

The 4,840.93 MiB serialized external-data file closely matches the 4,840.96 MiB
BFC region capacity created during initialization. The separate 3,792.80 MiB of
direct reserves primarily represents persistent prepacked buffers. After
warmup, `Shrink()` reclaimed 4,089.80 MiB of completely free BFC regions.
Internal fragmentation was only 248 bytes.

### Qwen 3.5 2B Text

Model:
`qwen3.5-2b-text-cuda-gpu:1`

- 24 decoder layers: 18 linear-attention and 6 full-attention layers
- 2 key/value heads and head size 256 for full-attention layers
- 187 CUDA `MatMulNBits` nodes
- 151 nodes declared nonzero workspace for the tested shape
- Weights embedded in the single `model.onnx` file

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| Planned workspace nodes | 0 | 151 | +151 |
| Largest workspace | 0 B | 6,952,960 B | +6,952,960 B |
| Serialized ONNX model | 1,397,119,352 B (1,332.40 MiB) | 1,397,119,352 B (1,332.40 MiB) | 0 B |
| Serialized external tensor data | 0 B | 0 B | 0 B |
| Post-initialization arena total | 3,248,802,560 B (3,098.30 MiB) | 3,248,802,560 B (3,098.30 MiB) | 0 B |
| Post-initialization direct reserved bytes | 1,343,619,072 B (1,281.38 MiB) | 1,343,619,072 B (1,281.38 MiB) | 0 B |
| Post-initialization BFC region capacity | 1,905,183,488 B (1,816.92 MiB) | 1,905,183,488 B (1,816.92 MiB) | 0 B |
| Post-initialization arena slack | 1,280,049,152 B (1,220.75 MiB) | 1,280,049,152 B (1,220.75 MiB) | 0 B |
| Shrink reclaimed after warmup | 1,788,608,512 B (1,705.75 MiB) | 1,788,608,512 B (1,705.75 MiB) | 0 B |
| Measured arena reservation | 708,753,664 B | 701,974,528 B | **-6,779,136 B** |
| Final arena slack | 708,753,664 B | 701,974,528 B | **-6,779,136 B** |
| Internal fragmentation | 8,116 B | 8,116 B | 0 B |
| Internal fragmentation ratio | 0.0004122% | 0.0004122% | 0 pp |
| Arena allocation calls | 13,945 | 8,358 | **-40.1%** |
| WDDM initialization peak | 3,848 MiB | 3,848 MiB | **0 MiB** |
| WDDM post-initialization usage | 3,426 MiB | 3,426 MiB | **0 MiB** |
| WDDM pre-inference usage | 2,294 MiB | 2,294 MiB | **0 MiB** |
| WDDM inference peak | 2,952 MiB | 2,944 MiB | **-8 MiB** |
| WDDM inference increase | 658 MiB | 650 MiB | **-8 MiB** |
| Average latency | 222.61 ms | 230.20 ms | +3.4% |
| P50 latency | 221.23 ms | 229.95 ms | +3.9% |
| P90 latency | 227.51 ms | 234.87 ms | +3.2% |
| Initialization | 100.47 s | 89.50 s | **-10.9%** |

Workspace preallocation reduced the controlled CUDA arena reservation by
6,779,136 bytes (approximately 6.47 MiB), close to the largest declared
workspace of 6,952,960 bytes (approximately 6.63 MiB). It eliminated 5,587
allocator calls and reduced the WDDM inference peak by 8 MiB.

The model's 1,332.40 MiB serialized ONNX file embeds its weights. Initialization
created 1,816.92 MiB of BFC region capacity and 1,281.38 MiB of persistent direct
reserves. After warmup, `Shrink()` reclaimed 1,705.75 MiB of completely free BFC
regions. Internal fragmentation was 8,116 bytes. The paired latency run was 3.4%
slower with preallocation; as with the other single paired measurements, this
does not establish a stable latency effect.

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
| Largest workspace | 0 B | 172,032 B | +172,032 B |
| Serialized ONNX graph | 522,204 B | 522,204 B | 0 B |
| Serialized external tensor data | 1,102,761,792 B (1,051.68 MiB) | 1,102,761,792 B (1,051.68 MiB) | 0 B |
| Post-initialization arena total | 2,191,218,944 B (2,089.71 MiB) | 2,191,218,944 B (2,089.71 MiB) | 0 B |
| Post-initialization direct reserved bytes | 964,689,920 B (920.00 MiB) | 964,689,920 B (920.00 MiB) | 0 B |
| Post-initialization BFC region capacity | 1,226,529,024 B (1,169.71 MiB) | 1,226,529,024 B (1,169.71 MiB) | 0 B |
| Post-initialization arena slack | 892,338,176 B (851.00 MiB) | 892,338,176 B (851.00 MiB) | 0 B |
| Shrink reclaimed after warmup | 1,445,957,632 B (1,378.97 MiB) | 1,445,957,632 B (1,378.97 MiB) | 0 B |
| Measured arena reservation | 536,984,064 B | 537,099,264 B | **+115,200 B** |
| Final arena slack | 536,984,064 B | 537,099,264 B | **+115,200 B** |
| Internal fragmentation | 440 B | 440 B | 0 B |
| Internal fragmentation ratio | 0.0000339% | 0.0000339% | 0 pp |
| Arena allocation calls | 19,191 | 10,903 | **-43.2%** |
| WDDM initialization peak | 2,406 MiB | 2,406 MiB | **0 MiB** |
| WDDM post-initialization usage | 2,406 MiB | 2,406 MiB | **0 MiB** |
| WDDM pre-inference usage | 1,658 MiB | 1,658 MiB | **0 MiB** |
| WDDM inference peak | 2,170 MiB | 2,170 MiB | **0 MiB** |
| WDDM inference increase | 512 MiB | 512 MiB | **0 MiB** |
| Average latency | 105.52 ms | 115.92 ms | +9.9% |
| P50 latency | 101.12 ms | 118.30 ms | +17.0% |
| P90 latency | 116.17 ms | 127.86 ms | +10.1% |
| Initialization | 6.87 s | 8.09 s | +17.8% |

Workspace preallocation did not reduce the controlled CUDA arena reservation for
this shape. It added 115,200 bytes, while eliminating 8,288 allocator calls. The
1,051.68 MiB serialized external-data file is context for the 1,169.71 MiB BFC
region capacity created during initialization. After initialization, 851.00 MiB
of that capacity was unused; after warmup, `Shrink()` reclaimed 1,378.97 MiB of
completely free BFC regions. The separate 920.00 MiB of direct reserves remained
allocated and primarily represents persistent prepacked buffers. Internal
fragmentation was only 440 bytes. WDDM process peaks were identical between
configurations.

### RTX 5090 summary

| Model | WDDM inference-peak change | Arena reservation change | Allocation-call change | Average-latency change |
|---|---:|---:|---:|---:|
| Qwen 2.5 1.5B | **-6 MiB** | **-4,003,328 B** | **-30.0%** | **-16.5%** |
| Qwen 2.5 7B | **0 MiB** | +530,432 B | **-30.0%** | -0.9% |
| Qwen 3.5 2B Text | **-8 MiB** | **-6,779,136 B** | **-40.1%** | +3.4% |
| Hy-MT2 1.8B | **0 MiB** | +115,200 B | **-43.2%** | +9.9% |

The memory benefit scales with the model's workspace requirement at this longer
sequence length. Qwen 2.5 reused approximately 3.82 MiB of activation storage,
and Qwen 3.5 reused approximately 6.47 MiB. Their process-scoped WDDM inference
peaks decreased by 6 MiB and 8 MiB, respectively. Qwen 2.5 7B and Hy-MT2 did not
produce net reservation reductions for the tested shape. All four models
substantially reduced allocator calls. Latency moved in different directions
across the single paired runs.

## NVIDIA T1000

The T1000 runs predate the initialization-breakdown instrumentation. Their
serialized-size, direct-reserve, arena-slack, and fragmentation metrics were not
recorded, so the original results are preserved below without inferred values.

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
