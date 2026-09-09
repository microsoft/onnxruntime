# CUDA Workspace Preallocation Benchmark Results

## Purpose

These local benchmarks compare the existing dynamic workspace allocation path with
run-scoped static workspace preallocation for CUDA `MatMulNBits`. The preallocated
path declares each kernel's workspace and includes its lifetime in ORT's activation
memory pattern.

The current RTX 5090 results approximate ONNX Runtime GenAI generation with a
1,024-token prefill followed by 128 batch-1 decode steps. Fixed-capacity CUDA
KV-cache tensors are shared between each layer's past inputs and present outputs.
The benchmark covers both fpA-intB CUTLASS workspace and legacy
dequantize-plus-cuBLAS workspace. The older T1000 prefill-only results are
retained separately as historical data.

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
| KV-cache capacity | 1,152 tokens |
| Cache progression | The same fixed-capacity CUDA tensors are bound as past inputs and preallocated present outputs |
| Attention mask | Fixed 1,152-token capacity; the valid prefix advances after each generated token |
| Output | Dynamically allocated logits plus preallocated, aliased present KV outputs |
| Dispatch comparison | fpA-intB (`ep.cuda.fpa_intb_gemm=1`) and legacy (`=0`) |
| Warmup | 2 complete scenarios |
| Memory measurement | 1 complete scenario |
| Timed measurement | 10 complete scenarios |
| Reported latency | End-to-end scenario, prefill phase, and per-token decode distributions, including 10% trimmed means and stall rates |
| Qwen 2.5 1.5B robust rerun | Six fresh-process pairs per path in three ABBA/BAAB blocks |

The tested Qwen packages declare `past_present_share_buffer: true`. Before each
complete scenario, the benchmark zeroes the fixed-capacity cache and resets the
attention mask. Every run verifies that each present output retains the fixed
shape and aliases the corresponding past-input device pointer. Prefill and
decode therefore reuse two stable memory-pattern shapes instead of creating a
larger KV tensor at every decode step.

One test invocation reports end-to-end, prefill, and decode timing for a
generation scenario. Each dispatch and preallocation configuration remains a
separate fresh process so arena regions and cached memory patterns from another
configuration cannot affect its measured high-water mark.

For robust latency analysis, the benchmark additionally reports:

- A **10% trimmed mean**, calculated after sorting samples and removing the
  lowest and highest 10%. With 10 complete scenarios, this removes one scenario
  from each end. Decode trimming operates on all 1,280 token samples.
- A **stall** is a sample greater than three times the median for that process.
  The benchmark reports the threshold, count, and rate separately for complete
  scenarios, prefill, and decode.
- Paired latency changes use adjacent scratch/preallocated processes within
  three order-balanced blocks: ABBA, BAAB, and ABBA. The median paired change
  summarizes the six comparisons, while the range exposes process-level
  instability that trimming cannot remove.

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

**Arena allocation calls** is the allocator's cumulative logical BFC allocation
count across initialization, warmup, memory measurement, and timed scenarios.
It includes requests served from existing arena regions and is not a
`cudaMalloc` count.

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

The detailed table records the original measurement pair. Because its latency
was noisy, the six-pair robust rerun below is the source of truth for latency
repeatability. Memory measurements were identical across reruns.

| Metric | Baseline | Preallocated | Difference |
|---|---:|---:|---:|
| **Generation, fpA-intB path** | | | |
| Planned workspace nodes | 0 | 141 | +141 |
| Largest workspace | 0 B | 6,291,456 B (6.00 MiB) | +6,291,456 B |
| Post-initialization arena total | 2,162,425,088 B | 2,162,425,088 B | 0 B |
| Post-initialization direct reserved bytes | 818,712,576 B | 818,712,576 B | 0 B |
| Post-initialization BFC region capacity | 1,343,712,512 B | 1,343,712,512 B | 0 B |
| Post-initialization arena slack | 818,712,576 B | 818,712,576 B | 0 B |
| Shrink reclaimed after warmup | 1,392,873,472 B | 1,392,873,472 B | 0 B |
| Measured arena reservation | 420,758,784 B (401.27 MiB) | 421,053,952 B (401.55 MiB) | +295,168 B (+0.07%) |
| Final arena slack | 420,758,784 B | 421,053,952 B | +295,168 B |
| Internal fragmentation | 15,139,064 B | 15,139,064 B | 0 B |
| Internal fragmentation ratio | 1.08767% | 1.08767% | 0 pp |
| Arena allocation calls | 150,704 | 149,041 | -1.1% |
| WDDM initialization peak | 2,742 MiB | 2,742 MiB | 0 MiB |
| WDDM post-initialization usage | 2,644 MiB | 2,644 MiB | 0 MiB |
| WDDM pre-inference usage | 1,896 MiB | 1,896 MiB | 0 MiB |
| WDDM inference peak | 2,300 MiB | 2,300 MiB | 0 MiB |
| WDDM inference increase | 404 MiB | 404 MiB | 0 MiB |
| End-to-end average | 1,452.68 ms | 937.10 ms | -35.5% (not repeatable) |
| End-to-end P50 | 1,434.68 ms | 918.61 ms | -36.0% (not repeatable) |
| End-to-end P90 | 1,574.73 ms | 962.70 ms | -38.9% (not repeatable) |
| Prefill average | 103.38 ms | 86.54 ms | -16.3% (not repeatable) |
| Decode average per token | 10.54 ms | 6.65 ms | -37.0% (not repeatable) |
| Decode P90 per token | 14.94 ms | 8.55 ms | -42.8% (not repeatable) |
| Decode P99 per token | 26.19 ms | 13.32 ms | -49.2% (not repeatable) |
| Initialization | 58.07 s | 70.86 s | +22.0% |
| **Generation, legacy path** | | | |
| Planned workspace nodes | 0 | 141 | +141 |
| Largest workspace | 0 B | 100,663,296 B (96.00 MiB) | +100,663,296 B |
| Post-initialization arena total | 1,343,712,512 B | 1,343,712,512 B | 0 B |
| Post-initialization direct reserved bytes | 0 B | 0 B | 0 B |
| Post-initialization BFC region capacity | 1,343,712,512 B | 1,343,712,512 B | 0 B |
| Post-initialization arena slack | 0 B | 0 B | 0 B |
| Shrink reclaimed after warmup | 674,776,064 B | 885,277,696 B | +210,501,632 B |
| Measured arena reservation | 521,422,080 B (497.27 MiB) | 449,889,536 B (429.05 MiB) | **-71,532,544 B (-13.7%)** |
| Final arena slack | 521,422,080 B | 449,889,536 B | **-71,532,544 B** |
| Internal fragmentation | 248 B | 248 B | 0 B |
| Internal fragmentation ratio | 0.0000180% | 0.0000180% | 0 pp |
| Arena allocation calls | 150,248 | 148,556 | -1.1% |
| WDDM initialization peak | 1,706 MiB | 1,706 MiB | 0 MiB |
| WDDM post-initialization usage | 1,706 MiB | 1,706 MiB | 0 MiB |
| WDDM pre-inference usage | 1,808 MiB | 1,808 MiB | 0 MiB |
| WDDM inference peak | 2,308 MiB | 2,238 MiB | **-70 MiB** |
| WDDM inference increase | 500 MiB | 430 MiB | **-70 MiB** |
| **Latency: median process metric across six fresh-process runs** | | | |
| End-to-end 10% trimmed mean | 837.51 ms | 849.19 ms | +1.4% |
| End-to-end P50 | 831.94 ms | 836.34 ms | +0.5% |
| Prefill 10% trimmed mean | 86.41 ms | 87.32 ms | +1.1% |
| Decode 10% trimmed mean per token | 5.77 ms | 5.82 ms | +0.9% |
| Decode P50 per token | 5.64 ms | 5.73 ms | +1.5% |
| Decode P90 per token | 6.94 ms | 7.01 ms | +1.1% |
| Decode P99 per token | 9.67 ms | 9.91 ms | +2.5% |
| Median decode stall rate | 0% | 0% | 0 pp |
| Initialization | 1.43 s | 1.49 s | +4.1% |

fpA-intB preallocation did not reduce the memory high-water mark. Legacy
preallocation reduced measured arena reservation by
71,532,544 bytes (68.22 MiB) and WDDM inference peak by 70 MiB.

The robust rerun produced six paired comparisons per path:

| Path and pair | Scratch trimmed mean | Preallocated trimmed mean | Trimmed-mean change | P50 change | Decode trimmed-mean change | Decode stalls, scratch / preallocated |
|---|---:|---:|---:|---:|---:|---:|
| fpA-intB 1 | 1,794.56 ms | 875.34 ms | -51.2% | -43.9% | -49.9% | 38 / 1 |
| fpA-intB 2 | 1,406.24 ms | 863.73 ms | -38.6% | -34.1% | -40.2% | 0 / 0 |
| fpA-intB 3 | 955.23 ms | 1,691.96 ms | +77.1% | +66.0% | +84.9% | 3 / 28 |
| fpA-intB 4 | 1,689.19 ms | 1,231.72 ms | -27.1% | -36.5% | -30.8% | 12 / 2 |
| fpA-intB 5 | 1,281.24 ms | 1,149.27 ms | -10.3% | -14.2% | -10.5% | 1 / 1 |
| fpA-intB 6 | 1,952.18 ms | 1,933.20 ms | -1.0% | -5.4% | -7.2% | 1 / 21 |
| **fpA-intB median paired change** | | | **-18.7%** | **-24.1%** | **-20.6%** | |
| Legacy 1 | 835.54 ms | 842.66 ms | +0.9% | -1.0% | +1.3% | 1 / 1 |
| Legacy 2 | 854.80 ms | 852.44 ms | -0.3% | +0.5% | -1.0% | 0 / 0 |
| Legacy 3 | 850.09 ms | 845.93 ms | -0.5% | -2.5% | -0.6% | 0 / 0 |
| Legacy 4 | 824.26 ms | 839.01 ms | +1.8% | +1.0% | +1.1% | 0 / 0 |
| Legacy 5 | 839.47 ms | 854.10 ms | +1.7% | +0.9% | +1.7% | 0 / 0 |
| Legacy 6 | 834.65 ms | 857.20 ms | +2.7% | +3.5% | +2.5% | 0 / 0 |
| **Legacy median paired change** | | | **+1.3%** | **+0.7%** | **+1.2%** | |

Legacy is repeatable: all six paired trimmed-mean changes are between -0.5% and
+2.7%, and the aggregate latency metrics in the main table differ by at most
2.5%, so preallocation has effectively no latency effect. fpA-intB remains
non-repeatable even after trimming: paired changes range from -51.2% to +77.1%.
Whole processes enter different performance modes, and an earlier dispatch
trace confirmed that fresh-process tactic profiling can select different
numbers of `M=1` GEMV and CUTLASS nodes. A median paired speedup is shown for
completeness but is not treated as a reliable preallocation effect. The
repeatable 1.5B result remains the legacy memory reduction.

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
| Shrink reclaimed after warmup | 4,259,151,872 B | 4,259,151,872 B | 0 B |
| Measured arena reservation | 573,047,040 B (546.50 MiB) | 573,047,040 B (546.50 MiB) | 0 B |
| Final arena slack | 573,047,040 B | 573,047,040 B | 0 B |
| Internal fragmentation | 12,681,464 B | 12,681,464 B | 0 B |
| Internal fragmentation ratio | 0.24601% | 0.24601% | 0 pp |
| Arena allocation calls | 198,960 | 149,041 | -25.1% |
| WDDM initialization peak | 9,650 MiB | 9,650 MiB | 0 MiB |
| WDDM post-initialization usage | 9,424 MiB | 9,424 MiB | 0 MiB |
| WDDM pre-inference usage | 5,710 MiB | 5,710 MiB | 0 MiB |
| WDDM inference peak | 6,260 MiB | 6,258 MiB | -2 MiB |
| WDDM inference increase | 550 MiB | 548 MiB | -2 MiB |
| End-to-end average | 1,642.61 ms | 1,706.61 ms | +3.9% |
| End-to-end P50 | 1,582.50 ms | 1,619.38 ms | +2.3% |
| End-to-end P90 | 1,809.19 ms | 1,793.15 ms | -0.9% |
| Prefill average | 267.46 ms | 269.18 ms | +0.6% |
| Decode average per token | 10.74 ms | 11.23 ms | +4.5% |
| Decode P90 per token | 11.98 ms | 13.31 ms | +11.1% |
| Decode P99 per token | 16.60 ms | 27.11 ms | +63.4% |
| Initialization | 128.42 s | 133.42 s | +3.9% |
| **Generation, legacy path** | | | |
| Planned workspace nodes | 0 | 141 | +141 |
| Largest workspace | 0 B | 234,881,024 B (224.00 MiB) | +234,881,024 B |
| Post-initialization arena total | 5,076,114,688 B | 5,076,114,688 B | 0 B |
| Post-initialization direct reserved bytes | 0 B | 0 B | 0 B |
| Post-initialization BFC region capacity | 5,076,114,688 B | 5,076,114,688 B | 0 B |
| Post-initialization arena slack | 0 B | 0 B | 0 B |
| Shrink reclaimed after warmup | 1,183,335,424 B | 1,259,881,472 B | +76,546,048 B |
| Measured arena reservation | 943,736,064 B (900.02 MiB) | 676,513,280 B (645.17 MiB) | **-267,222,784 B (-28.3%)** |
| Final arena slack | 943,736,064 B | 676,513,280 B | **-267,222,784 B** |
| Internal fragmentation | 248 B | 248 B | 0 B |
| Internal fragmentation ratio | 0.00000482% | 0.00000482% | 0 pp |
| Arena allocation calls | 150,248 | 148,556 | -1.1% |
| WDDM initialization peak | 5,490 MiB | 5,490 MiB | 0 MiB |
| WDDM post-initialization usage | 5,490 MiB | 5,490 MiB | 0 MiB |
| WDDM pre-inference usage | 5,668 MiB | 5,668 MiB | 0 MiB |
| WDDM inference peak | 6,572 MiB | 6,314 MiB | **-258 MiB** |
| WDDM inference increase | 904 MiB | 646 MiB | **-258 MiB** |
| End-to-end average | 1,627.66 ms | 1,617.01 ms | -0.7% |
| End-to-end P50 | 1,582.32 ms | 1,577.08 ms | -0.3% |
| End-to-end P90 | 1,726.39 ms | 1,674.60 ms | -3.0% |
| Prefill average | 294.60 ms | 293.75 ms | -0.3% |
| Decode average per token | 10.41 ms | 10.34 ms | -0.7% |
| Decode P90 per token | 11.28 ms | 11.07 ms | -1.9% |
| Decode P99 per token | 14.39 ms | 14.52 ms | +0.9% |
| Initialization | 10.85 s | 4.62 s | -57.4% |

fpA-intB preallocation did not reduce memory. Legacy preallocation reduced
measured arena reservation by 267,222,784 bytes
(254.84 MiB) and WDDM inference peak by 258 MiB. The arena change exceeds the
224 MiB largest individual workspace because memory-pattern placement also
changed BFC region packing and the allocation high-water mark. Legacy latency
was effectively unchanged.

### RTX 5090 summary

| Model | Workload and path | WDDM inference-peak change | Arena reservation change | Allocation-call change | Average-latency change |
|---|---|---:|---:|---:|---:|
| Qwen 2.5 1.5B | Shared-KV generation, fpA-intB | 0 MiB | +295,168 B | -1.1% | Six-pair range: -51.2% to +77.1%; not repeatable |
| Qwen 2.5 1.5B | Shared-KV generation, legacy | **-70 MiB** | **-71,532,544 B** | -1.1% | Six-pair median: +1.3%; effectively unchanged |
| Qwen 2.5 7B | Shared-KV generation, fpA-intB | -2 MiB | 0 B | **-25.1%** | +3.9% |
| Qwen 2.5 7B | Shared-KV generation, legacy | **-258 MiB** | **-267,222,784 B** | -1.1% | -0.7% |

The generation workloads show the strongest memory benefit when fpA-intB is
disabled and preallocation covers the legacy dequantized-weight workspace. The
legacy path reduced WDDM inference peak by 70 MiB for Qwen 2.5 1.5B and 258 MiB
for Qwen 2.5 7B. fpA-intB preallocation did not reduce the arena high-water mark
for either model. Fixed-capacity shared KV buffers also cut decode time
substantially relative to the previous dynamic-cache benchmark, which measured
about 19 ms per token for 1.5B and 33 ms for 7B. Those old results represented
KV-tensor replacement and are not mixed with the production-like shared-buffer
results above.

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
chunked dequantize plus cuBLAS GEMM. The other two families remain
workspace-free.

### Qwen 2.5 1.5B traced dispatch

An environment-gated diagnostic build traced every `MatMulNBits` invocation for
one 1,024-token prefill and one decode token. Scratch and preallocated processes
selected exactly the same implementations in this trace.

With fpA-intB enabled:

| Phase | Actual implementation | Nodes | Allocator-backed workspace |
|---|---|---:|---|
| Prefill (`M=1024`) | fpA-intB CUTLASS GEMM | 113 | **Yes: CUTLASS runner workspace** |
| Prefill (`M=1024`) | Legacy full dequantize + cuBLAS | 28 | **Yes: full dequantized weight matrix** |
| Decode (`M=1`) | fpA-intB CUDA GEMV | 113 | No |
| Decode (`M=1`) | Legacy fused 4-bit kernel + separate bias | 28 | No |

With fpA-intB disabled:

| Phase | Actual implementation | Nodes | Allocator-backed workspace |
|---|---|---:|---|
| Prefill (`M=1024`) | Full dequantize + cuBLAS | 140 | **Yes: full dequantized weight matrix** |
| Prefill (`M=1024`) | Chunked dequantize + cuBLAS | 1 | **Yes: one dequantized weight chunk** |
| Decode (`M=1`) | Fused 4-bit single-row kernel | 113 | No |
| Decode (`M=1`) | Fused 4-bit single-row kernel + separate bias | 28 | No |

The fpA-intB decode counts are an autotuned outcome, not a fixed model
property. Another fresh process selected CUDA GEMV for 84 fpA-eligible nodes
and CUTLASS GEMM for 29 nodes at `M=1`; those 29 nodes did request workspace.
Workspace mode itself does not select tactics, but scratch and preallocated
processes must use identical tactic distributions before their latency can
isolate the effect of workspace allocation.

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
