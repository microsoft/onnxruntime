# WebGPU Workspace Preallocation Benchmark Results

## Methodology

- Build: Release, Dawn Vulkan backend
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU
- Models: Qwen 2.5 1.5B and 7B, plus a Qwen 3 8B applicability check
- Workload: batch 1 cached prefill with a configurable number of new tokens and
  a one-token KV cache
- Warmup runs: 5
- Memory-measurement runs: 3
- Timed runs: 30
- Storage-buffer cache: disabled
- The primary 1,024-token results use three baseline and three workspace-only
  measurements in separate fresh processes, using the balanced order
  baseline/planned, planned/baseline, baseline/planned
- The long-context tables state the number of fresh processes per mode and use
  reversed ordering where more than one process per mode was run

The WebGPU benchmark is
`MatMulNBitsWorkspace.WebGpuQwen25WorkspacePreallocationBenchmark`. Set
`ORT_WEBGPU_WORKSPACE_BENCHMARK_MODEL` to `qwen2.5-1.5b`, `qwen2.5-7b`, or
`qwen3-8b`,
`ORT_WEBGPU_WORKSPACE_BENCHMARK_MODEL_PATH` to the model,
`ORT_WEBGPU_WORKSPACE_BENCHMARK_SEQUENCE_LENGTH` to a positive token count
(default 1,024), and use
`ORT_WEBGPU_WORKSPACE_BENCHMARK_PREALLOCATION=0` or `1` to select the
configuration.

The allocator measurements cover WebGPU's general default device allocator,
including activations, outputs, and kernel workspaces. Persistent initializer
buffers use a separate read-only allocator and are not included in these figures.
The phase-local allocator high-water mark is updated synchronously by allocation
and free operations. WDDM local-memory usage is sampled every 5 ms through
`IDXGIAdapter3::QueryVideoMemoryInfo`. The long-context whole-run WDDM peak is
the maximum observed across initialization, each warmup, and measured inference.
For configurations that failed before those measurements completed,
`nvidia-smi` sampled device usage every 250 ms from an idle GPU. Those failed-run
peaks are lower bounds because the allocation that triggered the OOM never
became resident.

## Qwen 2.5 1.5B

| Metric | Baseline | Workspace-only planning | Difference |
|---|---:|---:|---:|
| Planned workspace nodes | 0 | 85 | +85 |
| Planned workspace slots | 0 | 85 | +85 |
| Aggregate declared workspace | 0 B | 693,108,736 B (661.00 MiB) | +661.00 MiB |
| Largest workspace slot | 0 B | 18,350,080 B (17.50 MiB) | +17.50 MiB |
| Workspace-pattern peak | 0 B | 18,350,336 B (17.50 MiB) | +17.50 MiB |
| Default device allocator calls, 3 runs | 1,293 | 1,041 | **-252 (-19.5%)** |
| Default device allocator calls per run | 431 | 347 | **-84 (-19.5%)** |
| Default device allocator peak bytes in use | 346,882,048 B (330.81 MiB) | 362,086,656 B (345.31 MiB) | +15,204,608 B (+14.50 MiB) |
| WDDM initialization peak, three runs | 1,571.11 MiB | 1,571.11-1,573.11 MiB | No material difference |
| WDDM post-initialization usage, three runs | 1,350.36 MiB | 1,350.36-1,352.36 MiB | No material difference |
| WDDM pre-inference usage, three runs | 1,985.48-2,017.48 MiB | 2,026.05-2,158.05 MiB | No stable direction |
| WDDM inference peak, three runs | 2,080.98-2,104.98 MiB | 2,124.55-2,389.05 MiB | No stable direction |
| WDDM inference increase, three runs | 87.50-95.50 MiB | 52.50-296.00 MiB | No stable direction |
| Initialization latency, median process | 2,125.59 ms | 2,096.74 ms | -1.4% |
| Average latency, median process | 243.69 ms | 220.07 ms | **-9.7%** |
| P50 latency, median process | 240.08 ms | 215.39 ms | **-10.3%** |
| P90 latency, median process | 272.62 ms | 237.56 ms | **-12.9%** |
| P99 latency, median process | 310.21 ms | 290.73 ms | **-6.3%** |

Workspace-only planning replaces per-node workspace allocations with one
backing-buffer allocation per inference. The net reduction of 84 default-device
allocation calls per run is consistent with replacing 85 workspace-slot
allocations with one backing buffer. Because ordinary WebGPU activation memory
patterns remain disabled, the 17.50 MiB workspace-pattern buffer cannot overlap
activation storage. The measured default-device allocator peak consequently
increased by 14.50 MiB.

The aggregate 661.00 MiB declaration is not allocated simultaneously. The
workspace memory pattern reuses offsets according to kernel lifetimes, reducing
the required backing buffer to 17.50 MiB.

WDDM inference residency was not repeatable between fresh processes and changed
direction when the process order was reversed. These driver-managed residency
figures should therefore not be attributed to workspace planning. The allocator
measurements were identical in both process orders and provide the controlled
comparison.

The three baseline process-average latencies were 243.69, 242.71, and 275.02 ms.
The three workspace-only process averages were 220.07, 210.60, and 237.25 ms.
Using the median process result limits the influence of the slower third pair.
Workspace-only planning improved the median average, P50, and P90 latency in all
three balanced fresh-process comparisons.

## Qwen 2.5 7B

| Metric | Baseline | Workspace-only planning | Difference |
|---|---:|---:|---:|
| Planned workspace nodes | 0 | 85 | +85 |
| Planned workspace slots | 0 | 85 | +85 |
| Aggregate declared workspace | 0 B | 1,504,706,560 B (1,435.00 MiB) | +1,435.00 MiB |
| Largest workspace slot | 0 B | 38,797,312 B (37.00 MiB) | +37.00 MiB |
| Workspace-pattern peak | 0 B | 38,797,568 B (37.00 MiB) | +37.00 MiB |
| Default device allocator calls, 3 runs | 1,293 | 1,041 | **-252 (-19.5%)** |
| Default device allocator calls per run | 431 | 347 | **-84 (-19.5%)** |
| Default device allocator peak bytes in use | 384,950,272 B (367.12 MiB) | 416,407,808 B (397.12 MiB) | +31,457,536 B (+30.00 MiB) |
| WDDM initialization peak, three runs | 5,152.92 MiB | 5,152.92 MiB | 0 MiB |
| WDDM post-initialization usage, three runs | 4,864.55 MiB | 4,864.55 MiB | 0 MiB |
| WDDM pre-inference usage, three runs | 5,725.23-5,910.23 MiB | 6,289.30-6,994.30 MiB | No controlled conclusion |
| WDDM inference peak, three runs | 6,003.23-6,204.23 MiB | 6,994.36-7,200.30 MiB | No controlled conclusion |
| WDDM inference increase, three runs | 278.00-307.00 MiB | 0.06-911.00 MiB | No stable direction |
| Initialization latency, median process | 7,663.43 ms | 6,165.76 ms | -19.5% |
| Average latency, median process | 473.93 ms | 400.85 ms | **-15.4%** |
| P50 latency, median process | 472.30 ms | 396.69 ms | **-16.0%** |
| P90 latency, median process | 504.94 ms | 430.34 ms | **-14.8%** |
| P99 latency, median process | 521.37 ms | 457.55 ms | **-12.2%** |

The 7B workspace pattern reduces 1,435.00 MiB of aggregate per-node
declarations to a 37.00 MiB backing buffer. As with the 1.5B model, replacing
the 85 per-node workspace allocations with one backing allocation accounts for
the net reduction of 84 default-device allocation calls per inference. Because
the backing buffer cannot overlap ordinary WebGPU activations, the controlled
allocator peak increased by 30.00 MiB.

The baseline process-average latencies were 398.94, 479.45, and 473.93 ms. The
workspace-only process averages were 400.57, 400.85, and 419.69 ms. The first
pair was effectively neutral (+0.4%); the other two pairs improved by 16.4% and
11.4%. The median process result improved by 15.4%, but the large between-process
variation means this should be treated as directional rather than a precise
speedup estimate.

WDDM inference residency was consistently higher for the planned processes, but
the hundreds-of-MiB difference is far larger than the controlled 37.00 MiB
workspace buffer and varied substantially between runs. The exact allocator
high-water mark is the reliable memory comparison; no feature-attributable WDDM
delta is claimed.

## Long-context scaling and capacity

`Runs/mode` is the number of fresh processes for each configuration. For
example, three runs/mode means three baseline and three planned processes. Each
process still contains five warmups, three memory-measurement runs, and 30 timed
runs. The tables report the median process value; with two processes, that is
the midpoint of the two results. Single-process-per-mode comparisons are
preliminary.

### Successful workloads

| Model | New tokens | Runs/mode | Average latency: baseline / planned | P50: baseline / planned | P90: baseline / planned | P99: baseline / planned |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 8,192 | 2 | 1,904.38 / 1,761.00 ms (**-7.5%**) | 1,902.65 / 1,740.03 ms | 1,972.00 / 1,836.91 ms | 2,005.22 / 1,933.06 ms |
| Qwen 2.5 1.5B | 10,240 | 1 | 2,375.76 / 2,305.65 ms (**-3.0%**) | 2,383.76 / 2,302.69 ms | 2,425.77 / 2,349.66 ms | 2,464.66 / 2,373.53 ms |
| Qwen 2.5 7B | 4,096 | 3 | 1,707.81 / 1,690.61 ms (**-1.0%**) | 1,705.59 / 1,681.66 ms | 1,750.74 / 1,737.67 ms | 1,791.88 / 1,793.44 ms |
| Qwen 2.5 7B | 4,608 | 1 | 1,962.51 / 1,915.41 ms (**-2.4%**) | 1,956.99 / 1,919.21 ms | 2,004.38 / 1,952.92 ms | 2,029.38 / 1,973.93 ms |

The 8K 1.5B workload used two order-balanced pairs. Planning improved average
latency by 9.8% in baseline/planned order and by 5.1% in planned/baseline order.
The other extended workload with three runs/mode, 7B at 4K, was effectively
latency-neutral. The 10K 1.5B and 4.5K 7B comparisons have only one process per
mode and need repetition before treating their latency differences as stable.

### Memory and allocations

| Model | New tokens | Whole-run WDDM peak: baseline / planned | Default allocator peak: baseline / planned | Planned workspace-pattern peak | Allocation calls: baseline / planned |
|---|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 8,192 | 15,103.8 / 14,015.8 MiB (**-1,088.0 MiB**) | 2,646.1 / 2,762.1 MiB | 140.0 MiB | 1,293 / 1,041 |
| Qwen 2.5 1.5B | 10,240 | 21,361.3 / 19,276.3 MiB (**-2,085.0 MiB**) | 3,307.6 / 3,452.6 MiB | 175.0 MiB | 1,293 / 1,041 |
| Qwen 2.5 7B | 4,096 | 19,604.8 / 18,420.8 MiB (**-1,184.0 MiB**) | 1,468.1 / 1,588.1 MiB | 148.0 MiB | 1,293 / 1,041 |
| Qwen 2.5 7B | 4,608 | 22,139.8 / 20,110.3 MiB (**-2,029.5 MiB**) | 1,651.6 / 1,786.6 MiB | 166.5 MiB | 1,293 / 1,041 |

Planning consistently removed 252 allocations across the three
memory-measurement runs, or 84 allocations per inference. It also increased the
controlled allocator peak because the reusable workspace backing buffer remains
logically live. The lower WDDM peaks therefore do not mean that the workspace
requires less logical memory. They indicate less driver-level residency under
these memory-intensive workloads, consistent with avoiding repeated transient
buffer allocation, deferred destruction, pooling, and fragmentation in
Dawn/Vulkan. Unlike the exact allocator measurements, WDDM residency remains
driver-managed and workload-dependent.

### OOM capacity boundary

| Model | New tokens | Baseline observed peak | Planned observed peak | Result |
|---|---:|---:|---:|---|
| Qwen 2.5 1.5B | 12,288 | >=22,697 MiB | >=23,519 MiB | Both OOM |
| Qwen 2.5 7B | 5,120 | >=23,354 MiB | >=22,579 MiB | Both OOM |
| Qwen 2.5 7B | 6,144 | >=23,604 MiB | >=23,641 MiB | Both OOM |
| Qwen 2.5 7B | 8,192 | >=23,640 MiB | >=23,664 MiB | Both OOM |
| Qwen 2.5 7B | 12,288 | >=23,683 MiB | >=23,683 MiB | Both OOM |

All failures reproduced from zero reported GPU usage in fresh processes and
failed during the first warmup with
`vkAllocateMemory failed with VK_ERROR_OUT_OF_DEVICE_MEMORY`. Workspace
planning did not extend the maximum runnable context. On this 24,463 MiB GPU,
the tested Qwen 2.5 7B capacity boundary is between 4,608 and 5,120 new tokens;
the Qwen 2.5 1.5B boundary is between 10,240 and 12,288 new tokens.

## Qwen 3 8B

The cached `qwen3-8b-cuda-gpu-2` model cannot provide a meaningful comparison
between the baseline and workspace-only configurations without changing its
`MatMulNBits` dispatch attributes. Its 253 `MatMulNBits` nodes have the
following distribution:

| Nodes | Bits | Block size | Accuracy level |
|---:|---:|---:|---:|
| 135 | 4 | 128 | 0 |
| 118 | 8 | 128 | 0 |

Neither workspace-capable WebGPU implementation is selected:

- Subgroup-matrix requires block size 32, but every node uses block size 128.
- DP4A requires accuracy level 4, but every node uses accuracy level 0.

The model therefore declares zero workspace nodes and slots. Enabling
workspace preallocation would not change allocation behavior, memory use, or
latency, so a balanced baseline/planned comparison was not run. The benchmark
reports this configuration as skipped instead of presenting a no-op comparison.

One unchanged-model baseline process completed successfully with five warmups,
three memory runs, and 30 timed runs:

| Metric | Baseline |
|---|---:|
| Planned workspace nodes | 0 |
| Planned workspace slots | 0 |
| Default device allocator calls, 3 runs | 900 |
| Default device allocator calls per run | 300 |
| Default device allocator peak bytes in use | 470,851,584 B (449.04 MiB) |
| Initialization latency | 14,917.9 ms |
| Average latency | 11,304.0 ms |
| P50 latency | 10,003.5 ms |
| P90 latency | 17,206.5 ms |
| P99 latency | 19,380.2 ms |

Changing the 4-bit nodes to accuracy level 4 would enable DP4A and make
workspace planning measurable, but that would benchmark a derived dispatch
configuration rather than the cached model as provided.

## Note

WebGPU needs a separate workspace-only memory-pattern path because its ordinary
activation memory patterns remain disabled: activation planning assumes
addressable pointers, while WebGPU allocations are opaque `WGPUBuffer` handles.

### Required updates

1. **Represent workspace as a buffer region**
   - Replace the raw workspace pointer concept with
     `{buffer, offset_bytes, size_bytes}`.
   - `buffer` may be an opaque `WGPUBuffer`; arithmetic is performed on the
     offset, not the handle.
   - Implemented in
     `include\onnxruntime\core\framework\workspace_requirement.h`.
2. **Preserve offsets through WebGPU tensors**
   - Non-owning workspace tensors must retain the allocation base and byte
     offset.
   - WebGPU bind groups, copies, uploads/downloads, tensor views, segmented
     bindings, and graph capture must bind the correct range.
   - Implemented primarily under `onnxruntime\core\providers\webgpu\`.
3. **Support multiple workspace slots**
   - Each kernel can declare independently-lived temporary buffers with stable
     slot IDs.
   - Each slot receives a synthetic negative memory-pattern ID.
   - DP4A declares two slots: quantized activation and activation scales.
   - Subgroup MatMulNBits declares one conditional activation-prepack slot.
4. **Create a workspace-only planner**
   - When activation memory patterns are disabled, `ExecutionFrame` creates a
     separate `workspace_planner_`.
   - On the first successful run, it traces workspace allocation and release
     lifetimes. Kernels still dynamically allocate because no cached pattern
     exists yet.
   - On later runs, it allocates one provider-native backing buffer and returns
     each slot as a region within that buffer.
   - Ordinary WebGPU activations continue using normal dynamic allocations.
5. **Cache workspace patterns separately**
   - `SessionState` owns a separate session-wide workspace
     `MemoryPatternGroup`.
   - It is not keyed by runtime input shapes because declarations use static or
     maximum-shape upper bounds.
   - The first successful cache insertion wins, keeping cached pattern pointers
     stable.
6. **Acquire and release around kernel execution**
   - `OpKernelContextInternal` maps `(node, slot_id)` to the planned region.
   - Workspace lifetimes end when the kernel context is destroyed.
   - Missing patterns, unavailable backing buffers, or undersized regions fall
     back to dynamic allocation.
7. **Match declaration and dispatch logic**
   - Workspace must be declared only when the corresponding implementation
     will actually run.
   - Subgroup feature/configuration selection and DP4A eligibility use the same
     predicates during declaration and execution.
8. **Retain safety restrictions**
   - Workspace planning supports sequential execution only.
   - It is disabled for multiple logical streams on the same device because
     lifetime ordering is nondeterministic.
   - Requested alignment cannot exceed the allocator guarantee.
   - Failed execution does not cache a workspace pattern.

### MatMulNBits implementation count

The base WebGPU `ApplyMatMulNBits()` dispatcher has four implementation
families:

| Implementation | Per-run temporary buffers | Benefits from workspace planning |
|---|---|---|
| Subgroup-matrix | Conditional activation prepack | Yes, when `needsPrepack` is true |
| DP4A | Quantized activation and activation scales | Yes |
| Wide-tile | None | No current benefit |
| Generic tiled MatMulNBits | None | No current benefit |

Therefore, two of the four base MatMulNBits implementation families can
currently benefit.

DP4A has two multiplication variants--small-M and tiled--but both consume the
same two quantization workspaces, so they count as one high-level dispatch
family for workspace planning. Similarly, subgroup-matrix supports several
hardware tile configurations but remains one implementation family.

There are also three WebGPU operator kernels involving MatMulNBits:

1. `MatMulNBits`
2. `MatMulNBitsQkv`
3. `MatMulNBitsMlp`

Only the base `MatMulNBits` kernel currently declares workspace requirements.
The fused QKV and MLP kernels can internally call `ApplyMatMulNBits`, so they
are potential future beneficiaries, but they need their own declarations and
slot-lifetime handling. Thus, at the operator level, one of three is currently
wired for planned workspace; the other two could potentially be extended.
