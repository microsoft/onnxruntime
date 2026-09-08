# CUDA Workspace Preallocation Benchmark Results

## Purpose

These local benchmarks compare the existing dynamic workspace allocation path with
run-scoped static workspace preallocation for CUDA `MatMulNBits`. The preallocated
path declares each kernel's workspace and includes its lifetime in ORT's activation
memory pattern.

Most results below cover **prefill only**. Each of those runs processes a prefill
chunk with a one-token KV cache, which selects the fpA-intB GEMM path and its
CUTLASS workspace. The RTX 5090 prefill-only runs use 1,024 new tokens to
represent a long-prompt workload. The preserved T1000 runs used 64 new tokens.

The RTX 5090 section also includes a Qwen 2.5 1.5B generation scenario with a
1,024-token prefill followed by 128 chained batch-1 decode steps. It compares
preallocation for both the fpA-intB CUTLASS workspace and the legacy
dequantize-plus-cuBLAS workspace. Ordinary batch-1 decode typically selects a
workspace-free GEMV path, so the generation scenario reports prefill and decode
latency separately.

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

The generation scenario uses a separate methodology:

- Workload: one 1,024-token prefill followed by 128 chained decode steps
- Initial past key/value cache: empty
- Decode cache: each step's present key/value outputs become the next step's past
  key/value inputs
- Decode input IDs: deterministic and identical across configurations
- Warmup scenarios: 2
- Memory-measurement scenarios: 1
- Timed scenarios: 10
- Each scenario resets to the same initial empty cache
- Baseline and preallocated configurations ran in separate fresh processes
- fpA-intB comparison: `ep.cuda.fpa_intb_gemm=1`
- Legacy comparison: `ep.cuda.fpa_intb_gemm=0`

The generation timings include full logits and present key/value outputs. Each
complete warmup traverses every decode-cache length before measurement.

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

### Qwen 3 8B

Model:
`qwen3-8b-cuda-gpu:2`

- 36 decoder layers
- 8 key/value heads
- 253 CUDA `MatMulNBits` nodes
- 253 nodes declared nonzero workspace for the tested shape

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| Planned workspace nodes | 0 | 253 | +253 |
| Largest workspace | 0 B | 4,254,208 B | +4,254,208 B |
| Serialized ONNX graph | 485,603 B | 485,603 B | 0 B |
| Serialized external tensor data | 5,937,758,208 B (5,662.69 MiB) | 5,937,758,208 B (5,662.69 MiB) | 0 B |
| Post-initialization arena total | 11,305,310,464 B (10,781.58 MiB) | 11,305,310,464 B (10,781.58 MiB) | 0 B |
| Post-initialization direct reserved bytes | 5,367,709,696 B (5,119.05 MiB) | 5,367,709,696 B (5,119.05 MiB) | 0 B |
| Post-initialization BFC region capacity | 5,937,600,768 B (5,662.54 MiB) | 5,937,600,768 B (5,662.54 MiB) | 0 B |
| Post-initialization arena slack | 5,289,545,728 B (5,044.50 MiB) | 5,289,545,728 B (5,044.50 MiB) | 0 B |
| Shrink reclaimed after warmup | 5,289,545,728 B (5,044.50 MiB) | 5,289,545,728 B (5,044.50 MiB) | 0 B |
| Measured arena reservation | 638,632,448 B | 638,833,664 B | **+201,216 B** |
| Final arena slack | 638,632,448 B | 638,833,664 B | **+201,216 B** |
| Internal fragmentation | 248 B | 248 B | 0 B |
| Internal fragmentation ratio | 0.00000412% | 0.00000412% | 0 pp |
| Arena allocation calls | 21,573 | 12,212 | **-43.4%** |
| WDDM initialization peak | 11,690 MiB | 11,690 MiB | **0 MiB** |
| WDDM post-initialization usage | 11,116 MiB | 11,116 MiB | **0 MiB** |
| WDDM pre-inference usage | 6,176 MiB | 6,176 MiB | **0 MiB** |
| WDDM inference peak | 6,792 MiB | 6,792 MiB | **0 MiB** |
| WDDM inference increase | 616 MiB | 616 MiB | **0 MiB** |
| Average latency | 287.63 ms | 283.46 ms | **-1.5%** |
| P50 latency | 288.04 ms | 283.76 ms | **-1.5%** |
| P90 latency | 294.03 ms | 287.30 ms | **-2.3%** |
| Initialization | 123.02 s | 146.40 s | +19.0% |

Workspace preallocation did not reduce the controlled CUDA arena reservation for
this model and shape. It added 201,216 bytes (approximately 197 KiB), while
eliminating 9,361 allocator calls. The process-scoped WDDM inference peak was
unchanged. The 30-run latency distributions had no multi-second outliers and
were slightly faster with preallocation, although a single paired run does not
establish a stable latency effect.

The 5,662.69 MiB serialized external-data file closely matches the 5,662.54 MiB
BFC region capacity created during initialization. The separate 5,119.05 MiB of
direct reserves primarily represents persistent prepacked buffers. After
warmup, `Shrink()` reclaimed 5,044.50 MiB of completely free BFC regions.
Internal fragmentation was only 248 bytes.

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
| Qwen 3 8B | **0 MiB** | +201,216 B | **-43.4%** | **-1.5%** |
| Hy-MT2 1.8B | **0 MiB** | +115,200 B | **-43.2%** | +9.9% |

The memory benefit scales with the model's workspace requirement at this longer
sequence length. Qwen 2.5 reused approximately 3.82 MiB of activation storage,
and Qwen 3.5 reused approximately 6.47 MiB. Their process-scoped WDDM inference
peaks decreased by 6 MiB and 8 MiB, respectively. Qwen 2.5 7B, Qwen 3 8B, and
Hy-MT2 did not produce net reservation reductions for the tested shape. All five
models substantially reduced allocator calls. Latency moved in different
directions across the single paired runs.

### Qwen 2.5 1.5B generation scenario

Model:
`qwen2.5-1.5b-instruct-cuda-gpu:4`

Each scenario performs one 1,024-token prefill followed by 128 autoregressive
decode steps. The fpA-intB configurations exercise CUTLASS workspace during
prefill and normally use fpA-intB GEMV during decode. Disabling fpA-intB forces
prefill through dequantize-plus-cuBLAS workspace, while decode normally uses the
legacy fused Q4 path.

#### fpA-intB path

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| Planned workspace nodes | 0 | 141 | +141 |
| Largest workspace | 0 B | 6,291,456 B (6.00 MiB) | +6,291,456 B |
| Measured arena reservation | 420,757,760 B (401.27 MiB) | 421,855,744 B (402.31 MiB) | +1,097,984 B (+0.3%) |
| Arena allocation calls | 431,834 | 433,854 | +0.5% |
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

The 6 MiB fpA-intB workspace did not reduce memory for this complete generation
scenario. The measured arena reservation increased by approximately 1.05 MiB,
and the WDDM inference peak increased by 10 MiB. The 128 decode steps use
workspace-free fpA-intB GEMV in the normal dispatch, so their growing cache and
output allocations dominate the scenario's memory high-water mark.

#### Legacy path

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| Planned workspace nodes | 0 | 141 | +141 |
| Largest workspace | 0 B | 100,663,296 B (96.00 MiB) | +100,663,296 B |
| Measured arena reservation | 545,014,016 B (519.77 MiB) | 450,691,328 B (429.81 MiB) | **-94,322,688 B (-17.3%)** |
| Arena allocation calls | 431,378 | 429,686 | -0.4% |
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

Legacy workspace preallocation reduced the measured arena reservation by
94,322,688 bytes (approximately 89.95 MiB) and the process-scoped WDDM inference
peak by 86 MiB. The reduction is close to the 96 MiB largest declared workspace,
showing that the dequantized-weight buffer substantially overlapped activation
storage instead of requiring a separate allocation.

These are single paired measurements on a laptop GPU; fpA-intB ran
baseline-first, while legacy ran preallocated-first. The end-to-end, prefill, and
decode latency differences are small and move in mixed directions across
percentiles, so they do not establish a stable latency effect. The legacy memory
reduction is the main result from this generation scenario.

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
