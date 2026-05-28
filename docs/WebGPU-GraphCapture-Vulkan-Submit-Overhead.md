# WebGPU Graph Capture: Vulkan Per-Submit Overhead and the Single-Submit Replay Fix

## Summary

On the WebGPU EP, enabling Graph Capture (GC) on the **Vulkan** Dawn backend regressed
decode throughput substantially (e.g. Qwen3‑1.7B on RTX 5060 Ti: ~96 → ~60 tps,
roughly −38%), while the same model on the **D3D12** Dawn backend improved with GC
(~73 → ~113 tps, +55%).

Root cause: in GC mode, `Replay()` was still calling `wgpuQueueSubmit` once per
16 dispatches (~9 Submits per decoded token). Vulkan's `vkQueueSubmit` is
significantly heavier than D3D12's `ExecuteCommandLists`, and unlike the non‑GC
path there is no CPU encode work in between to hide that cost.

Fix: in `Replay()` only, collapse all chunks into a single `wgpuQueueSubmit`.
The chunked-Submit policy is preserved on the non‑GC path because it provides
CPU/GPU pipelining there.

The fix is gated by an environment variable
`ORT_WEBGPU_GC_SINGLE_SUBMIT` (default: off) while it is being evaluated.

## Background

### Non-GC path

For each `Run()`, the WebGPU EP encodes shader dispatches into a `CommandEncoder`
on the CPU as ops execute. `WebGpuContext` tracks pending dispatches and
calls `Flush()` (which finalizes the encoder and calls `wgpuQueueSubmit`)
whenever the pending count reaches `max_num_pending_dispatches_`
(currently 16). A Qwen decode step is roughly 144 dispatches, so this yields
~9 Submits per token.

The intermediate Submits are intentional: while the GPU executes batch *N*,
the CPU is encoding batch *N+1*. Removing them (we tested bumping
`max_num_pending_dispatches_` to 8192) drops no‑GC throughput from
~96 tps to ~72 tps because the GPU sits idle waiting for the CPU to encode all
dispatches before any work starts.

### GC (Graph Capture) path

With GC enabled, command buffers are recorded once during the first run and
stored. `Replay()` resubmits those pre‑recorded command buffers on
subsequent runs — there is no per-op CPU encoding work.

Originally, `Replay()` mirrored the same "Submit every 16 dispatches" policy.
With ~9 Submits per replay and **no CPU work to overlap with**, the per‑Submit
driver cost is paid serially on every token.

## Why Vulkan suffers more than D3D12

`wgpuQueueSubmit` in Dawn translates to:

- **D3D12** — `ID3D12CommandQueue::ExecuteCommandLists`. A thin kernel-mode
  hand-off. Per-Submit overhead is small.
- **Vulkan** — `vkQueueSubmit`. Builds `VkSubmitInfo`, manages per-submit
  `VkSemaphore` / `VkFence` synchronization, takes the queue mutex, performs
  validation, and round-trips through the user-mode driver. Dawn additionally
  manages per-Submit serials, fence signaling, and resource-lifetime tracking
  for the Vulkan backend. On NVIDIA's Vulkan driver in particular, this is
  measurably heavier than D3D12's path.

At a target of ~100 tps with ~9 Submits per token, that's ~900
`vkQueueSubmit` calls per second. At hundreds of microseconds per call on
this driver, the cost is multiple ms per token — exactly the gap between the
60 tps and 100 tps observations.

## Measurements (RTX 5060 Ti, Qwen3-1.7B decode)

| Build         | Backend | GC=0    | GC=1    |
| ------------- | ------- | ------- | ------- |
| MainVulkan    | Vulkan  | 96 tps  | 60 tps  |
| MainD3D       | D3D12   | 73 tps  | 113 tps |
| single-submit | Vulkan  | 80 tps  | 100 tps |

The single‑submit experiment patched `Replay()` to skip the intermediate
`Flush()` calls. The GC regression on Vulkan disappeared.

## Fix

`Replay()` performs a single `wgpuQueueSubmit` per replay. The non-GC dispatch
loop is unchanged.

Behavior:

- GC mode: 1 Submit per `Replay()` instead of ~9.
- Non-GC mode: unchanged — still chunks every `max_num_pending_dispatches_`
  dispatches so CPU encoding overlaps GPU execution.
- Profiling carve-out: when timestamp queries are enabled inside passes, the
  intermediate flushes are preserved so per-pass timing is meaningful.
- GPU work itself is identical — same dispatches, same order, same barriers.
  Only the number of host-side `vkQueueSubmit` calls changes.

### Env var (transitional)

`ORT_WEBGPU_GC_SINGLE_SUBMIT` (default: off). Set to `1` to enable the
single‑Submit replay path while we are gathering data. Once accepted, the
behavior should become the default for the Replay path.

## Take-aways

- **Chunked Submits** ⇒ CPU/GPU pipelining. Valuable whenever the CPU is
  encoding the next batch.
- **Single Submit** ⇒ minimize driver overhead. Valuable when the CPU has
  no encoding work to do (i.e., GC `Replay()`).
- Per-Submit cost is backend-dependent; Vulkan is materially more expensive
  per submit than D3D12 on the drivers tested.
- Conclusion: chunking should be coupled to *whether there is CPU work to
  overlap with*, not applied uniformly.
