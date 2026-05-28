# WebGPU Graph Capture on Vulkan: WPR Investigation and NVIDIA Driver Findings

## TL;DR

Even after the single-submit `Replay()` fix
([WebGPU-GraphCapture-Vulkan-Submit-Overhead.md](./WebGPU-GraphCapture-Vulkan-Submit-Overhead.md)),
`enableGraphCapture=1` on the **Vulkan** Dawn backend still regresses Qwen3‑1.7B
decode by ~3× on NVIDIA. On the same machine, `enableGraphCapture=1` on the
**D3D12** Dawn backend speeds up decode by ~1.4×, as expected.

Windows Performance Recorder (WPR/xperf) profiling across all four
backend × GC combinations shows the regression lives in the NVIDIA Vulkan
**driver submission path** (kernel + UMD), not in Dawn or ORT.

## Methodology

- Harness: onnxruntime‑genai `benchmark_e2e.py`
  - `--reuse_generator -l 3000 -g 500 -w 1 -m -1 -i D:\models\Qwen3-1.7B
    --use_random_tokens --chat_template "{input}"`
  - 5 iterations for GC=0, 2 iterations for GC=1 (to keep ETLs < ~10 GB).
- GPU: NVIDIA GeForce RTX 5060 Ti (driver 591.86).
- For each condition we install the matching backend‑specific build
  (`MainVulkan` vs `MainD3D`), patch `genai_config.json`
  (`enableGraphCapture` and `ep.webgpuexecutionprovider.dawnBackendType`),
  then start WPR (`CPU + GPU` profiles, filemode) *before* launching
  Python so all image‑load events are captured.
- ETLs are exported with
  `xperf -tle -tti -o <csv> -a profile -detail` (lightweight per‑module
  sampling, no stack walks → no measurement noise from stack walker).
- The per‑token diff script is at `d:\temp\diff_profile_4way.py`; the
  orchestrator is `d:\temp\wpr_run_genai_ab.py`.

> **Pitfall worth calling out:** earlier runs appeared to show that D3D12+GC
> *also* regressed, ruling Vulkan out. That was wrong. The local Vulkan‑only
> ORT build silently ignores `dawnBackendType=D3D12` (the backend isn't
> compiled in) and falls back to Vulkan, so the "D3D12" conditions were
> actually running on Vulkan. The fix was to install a separate D3D‑enabled
> ORT build (`MainD3D\Release\Release\onnxruntime.dll` +
> `onnxruntime_pybind11_state.pyd` + `dxil.dll` + `dxcompiler.dll` +
> `d3dcompiler_47.dll`) before each D3D12 condition.

## Throughput results

| Backend | GC=0 (tps) | GC=1 (tps) | GC=1 / GC=0 |
| ------- | ---------: | ---------: | ----------: |
| Vulkan  | 52.65      | 16.17      | **0.31×** (3.26× slower) |
| D3D12   | 44.51      | **62.22**  | **1.40×** (faster, expected) |

## Per‑token CPU module weights

Top contributors, normalized by tokens generated. Modules ranked by peak
per‑token sample count across the four conditions.

| Module                     | vk_gc0 | **vk_gc1** | d3d_gc0 | d3d_gc1 | vk1/vk0 | d3d1/d3d0 |
| -------------------------- | -----: | ---------: | ------: | ------: | ------: | --------: |
| **ntoskrnl.exe** (kernel)  | 10,361 | **59,018** | 11,639  | 14,131  | **5.70×** | 1.21×     |
| **dxgkrnl.sys** (KMD shim) |    253 |  **1,744** |    207  |    265  | **6.90×** | 1.28×     |
| **nvlddmkm.sys** (NV KMD)  |    377 |  **2,444** |    320  |    265  | **6.48×** | 0.83×     |
| **dxgmms2.sys** (DXGI MM)  |    289 |  **1,707** |    300  |    371  | **5.90×** | 1.23×     |
| **nvoglv64.dll** (NV GL/Vk UMD) | 1,914 | **4,096** |     0 |     0 | **2.14×** | —         |
| watchdog.sys               |     19 |       142  |     21  |     20  |   7.42× | 0.98×     |
| onnxruntime.dll            |  5,476 |     6,438  |  5,801  |  5,007  |   1.18× | 0.86×     |
| onnxruntime-genai.dll      |  1,014 |       675  |    978  |    556  |   0.67× | 0.57×     |
| nvwgf2umx.dll (NV D3D UMD) |      0 |         0  |  3,897  |  4,333  |    —    | 1.11×     |
| D3D12Core.dll              |      0 |         0  |    992  |  1,127  |    —    | 1.14×     |

Units: CPU profiling samples per generated token. Higher = more CPU spent
in that DLL/driver per decoded token.

### What this says

1. **Above the WebGPU API, the work is the same.** `onnxruntime.dll` is
   essentially flat across all four conditions (5.0k–6.4k samples/tok), so
   ORT is not doing anything extra in the Vulkan+GC path.
2. **GC successfully reduces host work.** `onnxruntime-genai.dll` drops
   ~33% under GC on Vulkan and ~43% under GC on D3D12 — replaying a
   captured command list cuts user‑mode submission cost as designed.
3. **The kernel + NV driver layer explodes only on Vulkan+GC.** Per token:
   - `ntoskrnl.exe`: 5.7× more samples (Vulkan) vs 1.2× (D3D12)
   - `dxgkrnl.sys`: 6.9× vs 1.3×
   - `dxgmms2.sys`: 5.9× vs 1.2×
   - `nvlddmkm.sys`: 6.5× vs 0.83×
   - `nvoglv64.dll` (NV's combined GL/Vulkan UMD): 2.1×
4. **Same NVIDIA hardware, same Windows kernel, same Dawn build (minus
   backend‑specific compile flags).** The only thing the Vulkan path
   touches that D3D12 doesn't is `nvoglv64.dll` and the Vulkan WDDM
   submission code. The cost lives there.

Earlier stack‑resolved profiling (`xperf -a stack -butterfly` from a
`CPU.Verbose` capture) found `dxgkrnl.sys!McGenEventWrite_EtwWriteTransfer`
at **57.8% inclusive** on the Vulkan+GC run — the kernel is emitting an
ETW event for every GPU submit/resource transition during replay. The
volume of ETW chatter is itself a symptom of how much WDDM bookkeeping the
driver does per replay.

The ETL sizes corroborate this: for the same workload at 2 iterations,
Vulkan+GC produces a **10.2 GB** ETL versus only **2.6 GB** for D3D12+GC.

## Working hypothesis

> **Correction (2026‑05‑18):** An earlier draft of this section claimed
> the regression came from NVIDIA having "no fast path for re‑submitting
> a captured `VkCommandBuffer`." Reading
> `onnxruntime/core/providers/webgpu/webgpu_context.cc::Replay()`
> showed that this is not what ORT/Dawn actually does on the GC=1 path:
> every forward gets a **fresh** `wgpu::CommandEncoder` /
> `wgpu::ComputePassEncoder`, every captured command is re‑issued
> (`SetPipeline`, `SetBindGroup`, `DispatchWorkgroups`), and Dawn‑Vulkan
> therefore records a fresh `VkCommandBuffer` and submits it. So the CB
> handle is **not** reused across replays in either GC=0 or GC=1. The
> hypothesis below has been rewritten accordingly.

What is actually different between GC=0 and GC=1 at the Vulkan level is
not CB reuse but **binding stability**:

| | GC=0 | GC=1 |
|---|---|---|
| `wgpu::ComputePipeline` handles | same (cached) | same (cached) |
| `wgpu::BindGroup` handles | often new each forward (transient buffers churn) | **same handles every forward** (captured once) |
| Underlying `VkBuffer`s referenced | vary forward‑to‑forward | **identical every forward** |
| `VkDescriptorSet`s bound | vary | **identical every forward** |
| `VkCommandBuffer` recording | fresh | fresh |
| `vkQueueSubmit` count per token | one per forward | one per forward |

In GC=1, after the first few forwards the NV driver sees **the same
descriptor sets, the same `VkBuffer`s, and the same dispatch sequence
on every submit, for hundreds of submits in a row**. In GC=0 the
descriptor sets / VkBuffers churn because transient buffers get freed
and re‑allocated each forward.

The hypothesis is therefore: NVIDIA's Windows Vulkan KMD has a slow
path that is triggered (or fails to early‑out) precisely when a submit
re‑references resources that were live in the previous submit — for
example a residency / state‑tracking walk keyed on "is this resource
already current?" that goes quadratic, or a descriptor / barrier
validation path that ignores per‑resource version stamps. Whatever the
exact mechanism, it lives in the KMD / WDDM layer, not in the UMD CB
recorder.

### "Wouldn't the driver be just as slow without GC then?"

No. GC=0 and GC=1 issue the same number of `vkQueueSubmit` calls per
generated token (one per forward). The denominator is identical. If
the per‑submit driver cost were the same, the per‑token kernel sample
count would also be the same. Instead we see per‑token cost in
`ntoskrnl`/`nvlddmkm`/`dxgkrnl`/`dxgmms2` jump 5.7–6.9× — the driver
is doing **more kernel work per submit** in GC=1, and the only thing
that systematically changes between GC=0 and GC=1 from the driver's
point of view is the binding stability described above.

Evidence that this lives in the KMD, not the UMD:

- `nvoglv64.dll` (NV user‑mode Vulkan UMD) only ~doubles (2.1×) — it
  just records and forwards the CB, and the per‑forward CB content is
  the same in both modes.
- `nvlddmkm.sys`/`dxgkrnl.sys`/`dxgmms2.sys` (KMD + WDDM) all blow up
  6–7×.

D3D12 doesn't show this regression because its residency / state‑
tracking model is exposed to the app (`ID3D12Fence`, explicit residency)
and the driver's per‑submit hot path is engineered around the
common case of stable resource lists.

## Standalone repro attempts (negative)

Built a minimal Vulkan compute repro at `D:\vulkan_replay-repro\`
(raw Vulkan, no Dawn, no WebGPU) to try to reproduce the regression
outside of ORT. The program issues `K` compute dispatches per iter,
each binding one storage buffer via a pre‑allocated `VkDescriptorSet`,
and pipelines submits via a `VkFence` ring. Axes tried so far, all on
RTX 5060 Ti / driver 591.86 / Windows:

| Axis | What it tests | Result |
|---|---|---|
| `--mode stable\|rotating` | Whether re‑binding the SAME `VkBuffer`s + `VkDescriptorSet`s every iter (stable, mimics GC=1) is slower than rotating through a larger pool (rotating, mimics GC=0). | **No signal.** ≤8% difference at K=16; flat at K=64,128. |
| `--inflight 1..4` | Whether ORT's lack of per‑iter `vkQueueWaitIdle` matters. Replaces it with a `VkFence` ring of depth N. | Pipelining gives a uniform ~20% speedup but does not unlock stable‑vs‑rotating signal. |
| `--batched-submit 1..16` | Collapses N iters' worth of dispatches into ONE CB + ONE `vkQueueSubmit`, mimicking llama.cpp's `ggml_vk_submit` which accumulates many ops before a single submit. | **No signal.** All variants within 0.25% at K=128. Submit overhead is already negligible at this granularity. |

What this rules out: the regression is NOT explained by per‑iter CB
recording, by binding stability alone, by submission depth, or by
how many `vkQueueSubmit`s are issued per "forward." A minimal raw‑
Vulkan program faithfully reproducing all four of those axes cannot
trigger the slow path.

What this implies: the trigger is something Dawn‑specific in the
shape of the `VkSubmitInfo` / resource set that Dawn‑Vulkan emits in
GC=1 — likely interaction of Dawn's residency tracking, its
descriptor‑pool layout, sub‑allocated buffer memory layout, or the
exact set of pipeline barriers it inserts — not the raw "same handles
every submit" property.

## How to prove this is an NV bug (Dawn instrumentation)

The standalone repro is not enough. To pin this on NV we need a
deterministic comparison of "exactly what ORT submits in GC=0 vs GC=1"
plus a way to feed a recorded submit‑stream back to the driver without
Dawn or ORT in the loop. Suggested instrumentation, in order of
diagnostic value:

### 1. Submit‑shape diff (cheapest, most informative) — RESULT

**Implemented.** Patch lives in `MainVulkan/Release/_deps/dawn-src/src/dawn/native/vulkan/QueueVk.cpp`
(build‑dir only, not committed to ORT). Adds env‑gated logging around
`vkQueueSubmit` inside `Queue::SubmitPendingCommandsImpl`:

- `ORT_DAWN_SUBMIT_DIFF=1` enables it
- `ORT_DAWN_SUBMIT_DIFF_EVERY=N` controls per‑submit log cadence
- `ORT_DAWN_SUBMIT_DIFF_FILE=path` redirects output
- Per submit logs: `idx`, `vkQueueSubmit` CPU `usec`, `cbCount`,
  FNV‑1a digest of the `VkCommandBuffer` handle(s),
  `match`/`run` (= same handle digest as previous submit),
  wait/signal semaphore counts. Every 256 submits: window avg `usec`
  and `match_rate`.

A/B run on Qwen3‑1.7B genai `benchmark_e2e.py` (l=3000, g=500, w=1,
Vulkan‑Dawn backend, RTX 5060 Ti, driver 591.86):

| metric                         | `gc=0`   | `gc=1`   | ratio |
|---                             |---:      |---:      |---:   |
| decode tps                     | 116.6    |  68.5    | 0.59× |
| total `vkQueueSubmit` calls    | 60,416   | 35,072   | 0.58× |
| tokens emitted                 |  2,500   |  1,000   | —     |
| **submits per token**          | **24.2** | **35.1** | **1.45×** |
| `vkQueueSubmit` CPU p50/submit | 26 µs    | 25 µs    | 0.96× |
| `vkQueueSubmit` CPU p99/submit | 65 µs    | 59 µs    | 0.91× |
| window‑avg `usec` median       | 27.3 µs  | 25.1 µs  | 0.92× |
| CB‑handle reuse (`match_rate`) | 0.000    | 0.000    | —     |

**Key findings:**

1. **The "stable bindings" hypothesis is wrong at the CB‑handle layer.**
   `match_rate=0.000` in both modes. Dawn re‑records and re‑allocates a
   fresh `VkCommandBuffer` for every `Replay()`/`SubmitImpl`, so the NV
   driver never sees the same CB handle twice — there are no stable
   command‑buffer handles for it to "specialize" on.

2. **The cost is NOT inside `vkQueueSubmit`.** Per‑submit CPU time is
   essentially identical between GC=0 and GC=1 (even marginally lower
   under GC=1, likely fewer wait/signal semaphores). The 5.7–6.9× KMD
   blow‑up seen in WPR (`ntoskrnl` / `nvlddmkm` / `dxgkrnl` / `dxgmms2`)
   is **not** caused by inflated work inside the `vkQueueSubmit` call
   itself.

3. **The actual signal: 1.45× more submits per token under GC=1.**
   GraphCapture on the Vulkan‑Dawn path issues **more** submits per
   token than GC=0, not fewer. With per‑submit CPU cost roughly
   constant, the kernel‑mode work scales linearly with submit count —
   which matches the WPR per‑token blow‑up pattern. So the user‑mode
   suspicion ("driver got slower per submit") is wrong; the real
   regression is "ORT/Dawn‑Vulkan issues more submits per token under
   GC=1."

**Hypotheses for next round of instrumentation** (in priority order):

- Count `Queue::SubmitImpl` invocations per `wgpuQueueSubmit` (a single
  WebGPU submit can produce multiple Vulkan submits if Dawn splits
  CBs for sync). If `gc=1` splits more (e.g. due to barrier insertion
  or recordCommands re‑entry), that's the regression.
- Count distinct `wgpuQueueSubmit` callsites per ORT `Run()` from the
  WebGPU EP (`webgpu_context.cc::Flush`, `Replay`). If GC=1 calls
  `Flush` from a different path that produces more sub‑submits, that's
  the regression.
- Compare against D3D12: if D3D12+GC=1 shows **fewer** submits per
  token than D3D12+GC=0, the divergence is on the Dawn‑Vulkan side
  specifically.

### 1‑original. Submit‑shape diff (full design — superseded by result above)

Patch Dawn at `src/dawn/native/vulkan/QueueVk.cpp::Queue::SubmitImpl`
(the function that calls `vkQueueSubmit`) to log, for every submit:

- `VkCommandBuffer` handle and the list of distinct `VkBuffer`,
  `VkImage`, `VkDescriptorSet`, `VkPipeline`, `VkPipelineLayout`
  handles touched during the submit (Dawn already walks resource
  usages for residency — extend that walk).
- For each `VkBuffer`: the backing `VkDeviceMemory` handle, the
  sub‑allocation offset, and the size used.
- The set of `VkSemaphore`s waited on / signaled, and the fence.
- The CB byte size (`vkGetDeviceMemoryCommitment` or a custom
  recorder count of `vkCmd*` calls intercepted upstream).

Capture this for ~50 forwards in both GC=0 and GC=1. The expected
finding is that GC=1 has a strictly identical handle set after the
first 2–3 submits, while GC=0 has handles drifting forward‑to‑
forward as transient buffers churn. If that's what we see, the
binding‑stability hypothesis is confirmed at the Dawn/Vulkan layer.

### 2. Dawn `VK_LAYER_KHRONOS_validation` + sync‑validation

Run a single GC=1 forward with sync‑validation enabled
(`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`,
`VK_VALIDATION_FEATURES_ENABLE=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`).
This will surface any false‑sharing / hazard the NV driver may be
defensively scanning for. Even if it's silent, "silent on a stable
binding stream" is itself a data point.

### 3. NV‑specific CB introspection via `VK_EXT_debug_utils`

Wrap each captured ORT compute pass with
`vkCmdBeginDebugUtilsLabelEXT("ort.replay.iter=N")` (Dawn already
threads `vkCmdBeginDebugUtilsLabelEXT` for trace markers in
`CommandBufferVk.cpp`). Then capture with Nsight Graphics in
**Frame Profiler** mode. Nsight shows per‑pass GPU time AND
per‑pass driver overhead, broken down by NV's own categories
(residency lookup, descriptor binding, barrier insertion). If
GC=1 shows higher driver overhead than GC=0 for byte‑identical
CBs, that's the bug, on NV's own profiler.

### 4. Replay the captured submit stream raw (kills "it's Dawn" hypothesis)

The strongest evidence. Add a Dawn debug mode that, alongside #1,
serializes each `VkCommandBuffer`'s commands to disk via a wrapper
that intercepts every `vkCmd*` call (use a thin Vulkan layer
or patch `CommandBufferVk::RecordCommands`). Build a tiny replayer
that re‑allocates the exact same `VkBuffer` / `VkDescriptorSet` /
`VkPipeline` set, replays the captured commands into a fresh CB
each iter, and submits — bit‑for‑bit identical to what Dawn does
in GC=1 but with **zero Dawn code in the loop**. If the replayer
reproduces the 3.3× slowdown vs a "rotating" variant of itself, the
bug is unambiguously below Dawn (i.e., in `nvoglv64.dll` /
`nvlddmkm.sys`).

This is the artifact to attach to the NV bug report: a self‑
contained program + recorded stream that NV can run on any
RTX‑class GPU without ORT, Dawn, or genai.

### 5. KMD‑layer profiling

If #4 reproduces, run it under WPR with `dxgkrnl`+`Microsoft-Windows-DxgKrnl`
providers enabled. The per‑submit cost should show the same 5.7–6.9×
blow‑up in `nvlddmkm.sys` / `dxgkrnl.sys` / `dxgmms2.sys` /
`ntoskrnl.exe` that we saw in the original 4‑way A/B. Same kernel
signature → same bug.

## What we can do on the ORT side

We can't fix the NVIDIA driver, but we can sidestep it.

### Workaround #0 — Choose backend automatically (cheapest)

When `enableGraphCapture=1` is requested on Windows + NVIDIA, default
`dawnBackendType` to **`D3D12`** unless the user has explicitly asked for
Vulkan. This is the only change that recovers the full 1.4× speedup and
doesn't require new Dawn work.

Pseudo‑code in `webgpu_provider_factory.cc::CreateWebGpuExecutionProvider`:

```cpp
if (config.enable_graph_capture &&
    config.backend_type == WGPUBackendType_Undefined) {
  // Auto: prefer D3D12 on Windows where it's available, because the
  // NVIDIA Windows Vulkan driver pays a ~3× kernel/UMD cost on replay.
  config.backend_type = WGPUBackendType_D3D12;
}
```

Risks: dual code paths in production traces; users that hardcode
`dawnBackendType=Vulkan` for testing won't see the speedup. Mitigate by
logging a one‑line warning when GC is on and the active backend is
Vulkan + NVIDIA.

### Workaround #1 — Coalesce submits

Today, even with the single‑submit replay fix, each forward call does
`vkQueueSubmit` once. For decode (one token at a time) we could batch N
decode steps into a single submit if the user provides a fixed
`max_decode_steps` and KV‑cache layout. The kernel‑side cost scales with
the number of `vkQueueSubmit` calls more than with CB length, so 8×
fewer submits ≈ 8× less driver overhead on this path.

Cost: requires a new genai/runtime API and stable shapes across steps.
Probably too invasive for a first pass.

### Workaround #2 — Re‑record at submit time (rejected)

Initial idea: rebuild a fresh `VkCommandBuffer` per submit so the driver
takes its "freshly recorded" fast path. Rejected after looking at the
numbers: the GC=0 condition is already "fresh record + submit" every
token, and it costs ~13k kernel samples/token on Vulkan. The replay path
costs ~69k. So re‑recording at replay can at best reproduce the GC=0
result (~52 tps) — it cannot beat it. The only reason GC=1 is worse than
GC=0 is the driver's replay overhead; eliminating the replay just gets us
back to GC=0, not above it. D3D12+GC (62 tps) remains the real win.

### Workaround #3 — Bypass `dxgkrnl` ETW emission

The kernel‑side ETW cost is intrinsic to WDDM; user‑mode software can't
disable it. However we can avoid the per‑resource transition events by
keeping every captured resource in a single, broad memory pool
(`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` + dedicated allocation off) so
that the WDDM resource list seen at submit is short. Worth probing with
Dawn allocator settings (`kVulkanUseDedicatedAllocations`, etc.) but
hard to test without a Dawn patch.

### Workaround #4 — Try `VK_KHR_synchronization2` + reduced barriers

NVIDIA's Vulkan driver historically takes a longer path through
`nvlddmkm` for pipeline barriers without `VK_KHR_synchronization2`. Dawn
defaults aren't always optimal here. We already experimented with
`DAWN_SKIP_BUFFER_BARRIERS=1` and saw a crash; a more surgical attempt
would collapse barriers across the captured CB at recording time.

### Workaround #5 — Driver version gate

The numbers above are on driver 591.86. If a future NV driver fixes the
replay fast‑path, we can drop the auto‑D3D12 selection. Until then, add a
runtime warning so users know Vulkan+GC on NVIDIA is a known slow
combination.

## Per‑callsite Flush breakdown (Vulkan GC=0 vs GC=1)

After `vkQueueSubmit` instrumentation proved (a) Dawn re‑allocates fresh
VkCommandBuffers every replay (`match_rate = 0.000`) and (b) Dawn does
not split CBs into multiple VkSubmitInfos (`vk/wgpu = 1.000`), the
remaining mystery was *which* ORT‑side `Flush()` callsite issues the
extra ~10 submits/token under GC=1.

`webgpu_context.cc` was instrumented with a `FlushSite` enum
(`{kImplicit, kRunFull, kReplayChunk, kReplayTail, kCaptureBegin}`) and
per‑callsite atomic counters, gated by `ORT_WEBGPU_FLUSH_TRACE=1` /
`ORT_WEBGPU_FLUSH_TRACE_EVERY=N` / `ORT_WEBGPU_FLUSH_TRACE_FILE=path`.
The 4 internal `Flush()` callsites (`Run`, `Replay` chunk, `Replay`
tail, `CaptureBegin`) tag themselves via `SetNextFlushSite()`; the 4
external callers (`buffer_manager.cc:478,548`, `webgpu_kernel.cc:55`,
`webgpu_execution_provider.cc:785`) remain in the `kImplicit` bucket.

Same workload (`-l 3000 -g 500 -w 1`, `-r 5` for gc0 / `-r 2` for gc1):

| Site            | GC=0 total | GC=0 /tok | GC=1 total | GC=1 /tok | Δ /tok |
|-----------------|-----------:|----------:|-----------:|----------:|-------:|
| `run_full`      |      47949 |     19.18 |        133 |      0.13 | **−19.05** |
| `replay_chunk`  |          0 |      0.00 |      23760 |     23.76 | **+23.76** |
| `replay_tail`   |          0 |      0.00 |       1484 |      1.48 |  +1.48 |
| `capture_begin` |          0 |      0.00 |          0 |      0.00 |   0.00 |
| `implicit` (4 external) | 12467 | 4.99   |       9439 |      9.44 |  +4.45 |
| **total**       |  **60416** | **24.17** |  **34816** | **34.81** | **+10.64** |

(GC=0: 2500 tokens. GC=1: 1000 tokens.)

**Findings:**

1. **`replay_chunk` is the dominant culprit.** Under GC=1, the chunked
   flush inside `Replay()` (one `wgpuQueueSubmit` per
   `max_num_pending_dispatches_` commands) issues **23.76 submits per
   token** — *more* than the non‑captured `run_full` path it replaces
   (19.18/tok). I.e. graph capture, far from collapsing submits, is
   issuing **24% more submits per token of equivalent work** on this
   workload.
2. **`implicit` (external `Flush()`) nearly doubles** under GC=1
   (4.99 → 9.44/tok, +89%). One or more of the 4 external sites is
   firing more often on the replay path — most likely the
   `buffer_manager.cc` upload/download flushes consumed by feed/fetch
   handling around `Replay()`.
3. `replay_tail` and `capture_begin` are negligible (1.48/tok and 0).

**Implications for the order of work:**

- The bug is not in Dawn‑Vulkan; it is in `webgpu_context.cc::Replay()`
  (chunked flush is too aggressive) and in whichever external callsite
  blew up to 9.44/tok.
- The earlier "Dawn‑Vulkan splits CBs more under GC" hypothesis is
  permanently dead; we don't need to investigate Dawn submit batching.
- `ORT_WEBGPU_GC_SINGLE_SUBMIT=1` (which collapses all
  `replay_chunk` + `replay_tail` into a single submit) eliminates this
  count entirely (down to ~11/tok in the extended A/B), but on this
  workload it *hurt* end‑to‑end throughput at prompt=3000 — suggesting
  the right fix is not "one giant CB" but "fewer chunks of the right
  size" (raise `max_num_pending_dispatches_` for the Replay path, or
  switch from a per‑command‑count budget to a per‑dispatch‑cost
  budget).

**Artifacts:**

- `d:\temp\flush_trace_runs\vulkan_gc0.flushtrace.txt`
- `d:\temp\flush_trace_runs\vulkan_gc1.flushtrace.txt`
- `d:\temp\flush_trace_ab.py` (orchestrator)
- ORT‑side instrumentation: `onnxruntime/core/providers/webgpu/webgpu_context.cc`
  (`FlushSite` enum, `SetNextFlushSite`, `GetFlushTraceConfig`).

## Suggested order of work

1. **Land Workaround #0** behind a build flag / config knob. Highest
   value, lowest risk; recovers the regression on NV.
2. Re‑run the WPR A/B on Intel and AMD GPUs to confirm the issue is
   NV‑specific. If it is, the auto‑select can be conditioned on the
   detected vendor ID.
3. ~~**Dawn instrumentation step #1** (submit‑shape diff)~~ **DONE.**
   See "RESULT" subsection above. Outcome: ruled out the binding‑
   stability and per‑submit‑slowness hypotheses; revealed the real
   regression is **submits/token (1.45× higher under GC=1)**, not
   driver work per submit.
4. ~~**Dawn `wgpuSubmit` counter + ORT‑side per‑callsite `Flush()`
   counter.**~~ **DONE.** See "Per‑callsite Flush breakdown" subsection
   below. Outcome: Dawn‑Vulkan does *not* split CBs (`vk/wgpu = 1.000`
   in all conditions). The 1.45× submit‑count increase under GC=1
   lives entirely in the ORT `Replay()` chunked‑flush path
   (`replay_chunk` accounts for 23.76 submits/token vs `run_full`'s
   19.18 in GC=0 — chunked replay actually issues **MORE** submits
   per token of equivalent work than the non‑captured path) plus a
   near‑doubling of "implicit" external `Flush()` callers
   (4.99 → 9.44/tok).
5. **Next: localize which of the 4 external `Flush()` callers grows
   under GC=1.** Sub‑tag the `implicit` bucket (buffer_manager.cc:478,
   buffer_manager.cc:548, webgpu_kernel.cc:55,
   webgpu_execution_provider.cc:785) and re‑run.
6. **Then: shrink `replay_chunk`.** Either raise
   `max_num_pending_dispatches_` for the Replay path, or switch the
   chunked flush to a budget based on actual GPU work rather than
   command count. Validate that this also helps D3D12 (it should be
   neutral there — D3D12+GC=1 was already 1.40× *faster* than
   D3D12+GC=0 even at the current chunk size).
7. Run the same flush‑trace + submit‑diff A/B on the D3D12 build for
   confirmation.
6. Build **Dawn instrumentation step #4** (raw replayer of captured
   submit stream) only if steps #4–5 above don't fully explain the
   delta. The replayer is the artifact for the NV bug filing.
7. If a true driver bug remains after #4–6, file an NVIDIA driver bug
   with the WPR trace excerpt + replayer. Otherwise the fix lives in
   Dawn‑Vulkan (reducing per‑token submit count under GC).

## Artifacts (this investigation)

All files live in `d:\temp\wpr_runs\` (not committed):

- `genai_{vulkan,d3d12}_{gc0,gc1}.etl` — raw WPR captures
- `genai_{vulkan,d3d12}_{gc0,gc1}_profile.csv` — per‑module CPU breakdown
- `gc1_butterfly.txt` — stack‑resolved butterfly from the earlier
  `CPU.Verbose` Vulkan+GC capture

Driver / orchestrator scripts (also not committed):

- `d:\temp\wpr_run_genai_ab.py` — 4‑condition WPR orchestrator with
  per‑backend ORT build install.
- `d:\temp\diff_profile_4way.py` — per‑token module diff used to produce
  the table above.
- `d:\temp\parse_butterfly.py` — parses `xperf -butterfly` HTML output.
