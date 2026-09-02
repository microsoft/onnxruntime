# WebGPU Workspace Preallocation Benchmark Results

## Methodology

- Build: Release, Dawn Vulkan backend
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU
- Model: Qwen 2.5 1.5B, 4-bit `MatMulNBits`
- Workload: batch 1 cached prefill with 1,024 new tokens and a one-token KV cache
- Warmup runs: 5
- Memory-measurement runs: 3
- Timed runs: 30
- Storage-buffer cache: disabled
- Baseline and workspace-only configurations ran in separate fresh processes

The WebGPU benchmark is
`MatMulNBitsWorkspace.WebGpuQwen25WorkspacePreallocationBenchmark`. Set
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
| WDDM post-initialization usage, three runs | 1,350.36 MiB | 1,350.36-1,354.92 MiB | No material difference |
| WDDM pre-inference usage, three runs | 1,995.48-2,514.98 MiB | 2,072.05-2,120.05 MiB | No stable direction |
| WDDM inference peak, three runs | 2,089.48-2,676.48 MiB | 2,177.05-2,397.05 MiB | No stable direction |
| WDDM inference increase, three runs | 87.50-161.50 MiB | 105.00-280.00 MiB | No stable direction |
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
