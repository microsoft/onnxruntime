# Flash Next WebGPU source handoff

Experimental source snapshot, November 3, 2026. Companion GenAI branch:
`microsoft/onnxruntime-genai`, `sayanshaw/flash-next-webgpu-portable`.
Use both branches together; this is not an upstream-ready PR.

## What is included

- Previously uncommitted WebGPU changes, tests and documentation.
- Source reconstruction from the frozen 152-translation-unit H200 provider
  compile map, plus the later three-worker long-K dense, GPU argmax, UInt32
  Slice and QMoE overlays.
- New fused kernels, replay headers and kernel registrations in normal source
  locations. Models, binaries, caches and credentials are not included.
- `source-manifest.json`: source hashes/origins, header-resolution notes and
  the H200 reference environment. Local paths are provenance, not prerequisites.
- `research/`: original source transforms, graph rewrites, fixture generators,
  CPU tests and Linux qualification/benchmark controllers, including rejected
  and default-off experiments. This directory is research evidence, NOT an
  automatically portable Windows test suite.

Held experiments are retained as source, not all enabled or stacked together.
The branch is a consolidated source snapshot, not a byte-identical rebuild
of the measured H200 libraries. In particular, a normal full rebuild uses a
single shared-header view instead of the historical isolated object relinks.
Windows compilation, adapter/backend compatibility and RTX Spark performance
must be checked on the target. No Windows pass is claimed.

## First steps on RTX Spark / Windows

1. Inventory hardware/CPU architecture, NVIDIA driver, available RAM/VRAM,
   existing workloads, Visual Studio/CMake/Python and actual WebGPU adapter.
   Under RDP reject WARP/software adapters. Do not assume a GPU model from
   the machine's nickname.
2. Inspect only configs and ONNX metadata in `C:\models`, without external
   weights. Prefer `all_int4` for compatibility screening, but verify actual
   custom operators, model type and quantization. New RC CUDA/DFlash exports
   are not necessarily compatible with the older H200 WebGPU graph.
3. Build matching native ORT WebGPU and GenAI from these branches using their
   supported build scripts. Keep compilation parallelism bounded. Inspect
   Windows Dawn backend settings; do not assume Vulkan and D3D12 expose the
   same shader features or workgroup limits.
4. Run a tiny correctness/routing smoke test first. Verify heavy operators
   execute on WebGPU, preserving intentional CPU embedding/Engram placement.
   Fail explicitly for unsupported operators; do not silently use CPU QMoE,
   requantize the model or rewrite the supplied configs.
5. Apply only supported flags from the manifest, not blindly the whole H200
   environment. The three-worker long-K route requires 384 invocations and
   15360 shared bytes and is shape-specific.

The original benchmark entry points are
`research/benchmark_dense_long_k.py` and
`research/benchmark_gpu_slice.py`; they use Linux `resource`/`/proc` telemetry,
absolute historical paths and a frozen prompt pool. Do not run their Linux
launch controllers unchanged on Windows. Adapt platform-specific telemetry
explicitly and record unavailable counters, without changing timing semantics.
The ordinary GenAI setup registers the provider with
`onnxruntime_genai.register_execution_provider_library("webgpu", plugin_path)`.
Model/provider config and prompt fixture must be validated for the target.

## Protocol and honest comparison

Single sequence, greedy, MTP/speculation off. Start at 2K; only then 8K/32K.
One excluded warmup and three measured requests, 256 output tokens each.
Decode is 255 divided by elapsed time from first to last emitted token.
Prefill includes append_tokens through first-token selection. TTFT is request
start to first emission. Report P50/P95 and a disclosed sampled GPU peak,
including load/warmup, plus host peak and exact source/model/flag identities.

Use the same model, prompts, device/backend and settings for control/candidate.
Do not treat profiler GPU sums or a single kernel gain as model throughput.
Preserve generated-token checks; they are not independent whole-model quality.
Cold Engram page faults invalidate steady-state comparisons.

Historical H200 full-validation measurements:

| Context | Decode tok/s | Inclusive prefill tok/s | TTFT P50/P95 s | Sampled peak GiB |
|---|---:|---:|---:|---:|
| 2K | 83.00 | 652.02 | 3.141/3.144 | 68.96 |
| 8K | 81.63 | 600.20 | 13.629/13.729 | 69.30 |
| 32K, held/not repeat-qualified | 71.95 | 531.49 | 61.665/61.674 | 70.68 |

Best repeated 2K retained-validation decode was 84.11 tok/s. These are not
RTX Spark predictions. Tianlei's CUDA RC mixed-bit/speculative 1024-output
table is a different workload and cannot be relabeled as WebGPU performance.

Detailed experiments and limitations are in
`../../docs/contrib_ops/webgpu/qmoe_gate.md`.

Keep the workspace small. Do not index model/build directories, load enormous
profiles or start multiple model instances. Monitor editor/Copilot RSS as well
as RAM/VRAM; release task-owned allocations and disposable intermediates at
phase boundaries. Never disturb unrelated workloads.
