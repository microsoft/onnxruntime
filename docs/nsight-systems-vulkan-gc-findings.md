# Layer 1 — Nsight Systems Vulkan trace findings (WGPU9, GC=on vs GC=off)

This is the read-only trace step from the *Debugging the Vulkan + GC regression — work plan* in [`WebGPU-Vulkan-Backend-Windows.md`](../docs/WebGPU-Vulkan-Backend-Windows.md). Source artifacts under `tmp/vulkan_gc_trace/` (trace files git-ignored; CSVs included).

## Setup

- **ORT**: WGPU9 wheel + post-`DAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON` overlay (`onnxruntime_pybind11_state.pyd` / `onnxruntime.dll` timestamp 2026-05-13 14:32). Vulkan-only build.
- **Branch**: `hari/webgpu_vulkan_gc_fix` from `hari/webgpu_perf_1_full @ d5e208bffe`.
- **Model**: Qwen3-1.7B (4-bit MatMulNBits) at `D:\models\Qwen3-1.7B`.
- **Driver**: NVIDIA 591.86 on RTX 5060 Ti, Vulkan loader 1.4.341.1.
- **nsys**: 2025.6.3.343, `--trace=vulkan --vulkan-gpu-workload=batch --sample=none --cpuctxsw=none`.
- **Workload** (small, to keep traces ≤ 5 MB): `benchmark_e2e.py --reuse_generator -l 512 -g 50 -w 1 -r 2 -m -1 --use_random_tokens` → 3 prefill steps + 150 decode steps total.
- **Variants**: only `model.decoder.session_options.provider_options[].WebGPU.enableGraphCapture` differs between the two reports (`"1"` vs `"0"`); both runs use `dawnBackendType="Vulkan"`.

> The Vulkan implicit layer install requires admin once; we did this via `tmp/vulkan_gc_trace/trace_admin.ps1` running under UAC. After that, all `nsys` runs are non-elevated.

## Headline result

Graph capture on Vulkan does **not** reach a true zero-overhead replay. Per-iter Vulkan API counts and CPU time both rise sharply with GC enabled — driven entirely by Dawn allocating and freeing GPU memory inside what should be the replay path.

Per ~150 decode steps:

| Vulkan API | GC=on calls | GC=off calls | Δ calls | GC=on time | GC=off time | Δ time |
|---|---:|---:|---:|---:|---:|---:|
| `vkAllocateMemory` | **9 530** | 1 298 | **+7.3×** | 745 ms | 224 ms | **+521 ms** |
| `vkFreeMemory` | **9 530** | 1 298 | **+7.3×** | 1 286 ms | 402 ms | **+884 ms** |
| `vkCreateBuffer` | **18 464** | 1 741 | **+10.6×** | 9 ms | 2.5 ms | +6.5 ms |
| `vkBindBufferMemory` | **18 464** | 1 741 | **+10.6×** | 3.5 ms | 0.5 ms | +3.0 ms |
| `vkMapMemory` | **9 345** | 1 113 | **+8.4×** | 1.0 ms | 0.2 ms | +0.8 ms |
| `vkQueueSubmit` | 4 045 | 3 578 | +13 % | 140 ms | 115 ms | +25 ms |
| `vkBeginCommandBuffer` | 4 046 | 3 579 | +13 % | 15 ms | 13 ms | +2 ms |
| `vkEndCommandBuffer` | 4 045 | 3 578 | +13 % | 6.8 ms | 5.7 ms | +1 ms |
| `vkCmdPipelineBarrier` | 8 377 | (not in top) | — | 3.8 ms | — | — |
| `vkUpdateDescriptorSets` | 10 664 | **39 942** | −3.7× | 3.3 ms | 12.3 ms | −9 ms |
| `vkCmdBindPipeline` | 56 856 | 39 942 | +42 % | 19 ms | 13 ms | +6 ms |
| `vkWaitForFences` | 297 | 297 | 0 | 137 ms | 206 ms | **−68 ms** |
| `vkAllocateCommandBuffers` | 33 | 32 | +1 | 0.45 ms | 0.49 ms | ≈0 |
| `vkResetCommandBuffer` | (absent) | (absent) | — | — | — | — |

Numbers from `tmp/vulkan_gc_trace/stats_gc_on_vulkan_api_sum.csv` and `stats_gc_off_vulkan_api_sum.csv`. The "Total Time (ns)" columns are CPU time inside the Vulkan API call, not GPU time.

## Interpretation

1. **Allocation churn is the regression.** Combined `vkAllocateMemory` + `vkFreeMemory` CPU cost is **2.03 s** with GC vs **0.63 s** without — a **+1.4 s** CPU-side penalty over 150 decode tokens, i.e. **~9 ms/token of pure allocator overhead**. This matches the perf-matrix gap: Vulkan GC=on (72 tps ≈ 13.9 ms/tok) vs Vulkan GC=off (124 tps ≈ 8.1 ms/tok), a 5.8 ms/tok regression. The trace workload is smaller and noisier but the order of magnitude lines up.
2. **The replay does run.** `vkQueueSubmit` and `vkBeginCommandBuffer` only rise ~13 % between GC=off and GC=on — Dawn is *not* re-recording the whole graph from scratch every iteration. But it is allocating/freeing buffer-backing memory inside that replay.
3. **GPU side is fine.** `vkWaitForFences` total time is actually **lower** with GC (137 ms vs 206 ms). So the GPU finishes faster under GC; the regression is entirely on the CPU/driver side.
4. **`vkResetCommandBuffer` is absent in both traces** — Dawn isn't re-using a pool of command buffers via reset; it allocates fresh ones (33 / 32 `vkAllocateCommandBuffers` matches roughly the number of distinct command pools, not per-iter).
5. **`vkUpdateDescriptorSets` *drops* under GC** (40k → 11k). Consistent with capture-then-replay genuinely caching descriptor work. So the capture path *is* doing some of its job — just not for memory allocations.
6. **`vkCreateBuffer` + `vkBindBufferMemory` rise lockstep with `vkAllocateMemory`**, all at ~7.3-10.6×. So Dawn's per-replay alloc isn't a single big block being suballocated — each replay creates fresh `VkBuffer` objects, binds fresh memory, then later frees them. Classic per-frame transient buffer pattern.

## Hypothesis going into Layer 2

Dawn's Vulkan backend creates a new staging/upload buffer (or scratch memory for an internal copy) **per replay** of the captured command set, instead of reusing the buffer captured during the warmup pass. On D3D12 this is invisible because:
- D3D12 has implicit suballocator pooling at the placed-resource layer in Dawn,
- and/or DXGI's resource state tracking lets Dawn reuse a single buffer across replays.

On Vulkan, Dawn's `ResourceMemoryAllocator` is going through `vkAllocateMemory` directly (no pool) and the per-replay path doesn't tag the buffer as cacheable.

This will be confirmed (or rejected) in Layer 2 by adding instrumentation in `WebGpuContext::Run()` and the GC submit shim, and by tagging which Dawn call site is producing the alloc burst. The most likely Dawn call sites are:

- `BackendVk.cpp` / `ResourceMemoryAllocatorVk.cpp` — the per-allocation path.
- The GC implementation in `Device.cpp` / `Queue.cpp` for replay submission.

## Reproduction

```powershell
# Pre-req: Vulkan implicit layer registered. One-time admin run:
Start-Process powershell -Verb RunAs -ArgumentList `
    '-NoProfile','-File','d:\onnxruntime2\tmp\vulkan_gc_trace\trace_admin.ps1','-Gc','on'

# After that, normal terminal:
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe'
cd D:\onnxruntime-genai\benchmark\python
foreach ($gc in 'on','off') {
    # Patch genai_config.json (no BOM allowed -- genai parser rejects BOM):
    $cfgPath = 'D:\models\Qwen3-1.7B\genai_config.json'
    $cfg = Get-Content $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($po in $cfg.model.decoder.session_options.provider_options) {
        if ($po.PSObject.Properties.Name -contains 'WebGPU') {
            $po.WebGPU.enableGraphCapture = if ($gc -eq 'on') {'1'} else {'0'}
        }
    }
    [System.IO.File]::WriteAllText($cfgPath, ($cfg | ConvertTo-Json -Depth 20),
                                   (New-Object System.Text.UTF8Encoding $false))

    & $nsys profile --trace=vulkan --vulkan-gpu-workload=batch `
                    --sample=none --cpuctxsw=none --force-overwrite=true `
                    -o "d:\onnxruntime2\tmp\vulkan_gc_trace\trace_gc_$gc" `
                    C:\Users\hasesh\AppData\Local\Programs\Python\Python312\python.exe `
                    benchmark_e2e.py --reuse_generator -l 512 -g 50 -w 1 -r 2 -m -1 `
                                     -i D:\models\Qwen3-1.7B `
                                     --use_random_tokens --chat_template "{input}"
    & $nsys stats --report vulkan_api_sum --format csv --force-export=true `
                  --output "d:\onnxruntime2\tmp\vulkan_gc_trace\stats_gc_$gc" `
                  "d:\onnxruntime2\tmp\vulkan_gc_trace\trace_gc_$gc.nsys-rep"
}
```

## Caveats

- This trace deliberately uses a tiny workload (prompt 512 / gen 50) to keep the `.nsys-rep` files small (~3.6 MB GC=on, similar GC=off) and `nsys stats` fast. Repeating with the canonical workload (prompt 3000 / gen 500 / 5 runs) is in the work plan but optional unless Layer 2 disagrees on the call-count ratios.
- The 297 `vkWaitForFences` calls in both traces look identical — this is fence creation/teardown for command-pool flushes during model load, not steady-state decode fences. Dawn appears to handle steady-state decode synchronization through `vkQueueSubmit` + implicit fence batching, not explicit waits.
- `vkCmdPipelineBarrier` is only listed in the GC=on table — it does appear in GC=off too but below the auto-truncated top-N. Not a regression.
