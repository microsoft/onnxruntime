# WebGPU Workspace Preallocation Benchmark Results

## Methodology

- Build: Release, Dawn Vulkan backend
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU
- Models: Qwen 2.5 1.5B and 7B, 4-bit `MatMulNBits`
- Workload: batch 1 cached prefill with 1,024 new tokens and a one-token KV cache
- Warmup runs: 5
- Memory-measurement runs: 3
- Timed runs: 30
- Storage-buffer cache: disabled
- Three baseline and three workspace-only measurements ran in separate fresh
  processes, using the balanced order baseline/planned, planned/baseline,
  baseline/planned

The WebGPU benchmark is
`MatMulNBitsWorkspace.WebGpuQwen25WorkspacePreallocationBenchmark`. Set
`ORT_WEBGPU_WORKSPACE_BENCHMARK_MODEL` to `qwen2.5-1.5b` or `qwen2.5-7b`,
`ORT_WEBGPU_WORKSPACE_BENCHMARK_MODEL_PATH` to the model and use
`ORT_WEBGPU_WORKSPACE_BENCHMARK_PREALLOCATION=0` or `1` to select the
configuration.

The allocator measurements cover WebGPU's general default device allocator,
including activations, outputs, and kernel workspaces. Persistent initializer
buffers use a separate read-only allocator and are not included in these figures.
The phase-local allocator high-water mark is updated synchronously by allocation
and free operations. WDDM local-memory usage is sampled every 5 ms through
`IDXGIAdapter3::QueryVideoMemoryInfo`.

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
