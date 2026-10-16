# CPU offload and adaptive expert placement for MoE and QMoE

The built-in CUDA execution provider can execute selected experts on CPU while keeping the remaining experts on
CUDA. Both static placement and asynchronous, counter-driven expert exchanges are implemented for
`com.microsoft::MoE` and `com.microsoft::QMoE`. The graph nodes remain assigned to CUDA: CPU offload is internal to
their kernels, not execution-provider fallback or a graph rewrite.

The implementation and measurements below accompany [PR #33347](https://github.com/microsoft/onnxruntime/pull/33347).
The measured model is **Qwen3.5-35B-A3B**.

## Supported configurations

| Operator | Supported CPU offload |
|---|---|
| `MoE` | FP16/BF16 activations and expert weights, with constant FC1/FC2 weights and connected biases. |
| `QMoE` | `quant_type="int"`, uniform INT4 or INT8 FC1/FC2 weights, FP16/BF16 activations and scales, and canonical `weights_prepacked=0` initializers. |

Both require a statically known hidden dimension and enabled prepacking. The CPU expert count is global across the
session's eligible nodes; it is not a per-layer count. A node may have all, some, or none of its experts on CPU.
Setting the count to zero leaves the existing execution path unchanged.

Unsupported offload configurations include FP32 MoE, FC3 inputs, sparse mixer, FP4/FP8/INT2 QMoE, mixed-width or
externally prepacked QMoE weights, minimal builds, the plugin CUDA EP, CUDA Graph capture, and overlapping runs on
one configured session. Multi-device expert placement and redistribution of CUDA slots between nodes are not
implemented. These restrictions concern CPU offload, not the operators' ordinary all-GPU support.

## Session configuration

| Session option | Default | Meaning |
|---|---|---|
| `session.moe_cpu_offload_experts` | `0` | Global CPU-resident expert count; integer from zero through the number of eligible experts. |
| `session.moe_cpu_offload_strategy` | `adaptive` | `adaptive` retains popularity-based placement and exchanges. QMoE-only `basic` fills earlier nodes on GPU and fixes placement. |
| `session.moe_expert_counter_alpha` | `0.75` | Finite counter decay coefficient in `[0, 1]`. |
| `session.moe_expert_counter_probability_weight` | `0` | Finite blend in `[0, 1]` between binary selection and routing probability. |
| `session.moe_expert_swap_epsilon` | `0.01` | Finite, non-negative relative margin required for an exchange. |
| `session.moe_max_in_flight_swaps_per_device` | `4` | Non-negative exchange limit per CUDA device; `0` keeps placement static. |
| `session.moe_expert_counter_state_file` | Unset | Input JSON containing `alpha`, `epsilon`, and per-expert popularities. |
| `session.moe_expert_popularity_file` | Unset | Output JSON written when the session is destroyed. |

For example, configure an ordinary CUDA session with 4,096 CPU experts and at most five adaptive exchanges:

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_session_config_entry("session.moe_cpu_offload_experts", "4096")
options.add_session_config_entry("session.moe_expert_counter_alpha", "0.9")
options.add_session_config_entry("session.moe_expert_swap_epsilon", "0.1")
options.add_session_config_entry("session.moe_max_in_flight_swaps_per_device", "5")

session = ort.InferenceSession("model.onnx", options, providers=["CUDAExecutionProvider"])
```

These are benchmark settings, not runtime defaults.

## Current implementation

### Initial placement and ownership

The session owns the global counters, initial placement policy, and a `KernelPilot` for each registered kernel.
After registration and optional state loading, it assigns:

```text
CUDA expert count = eligible expert count - configured CPU expert count
```

With the default `adaptive` strategy and nonzero saved counters, the highest-ranked eligible experts receive CUDA slots. Ties use local expert ID,
then graph scope and node index. If all scores are zero, CUDA slots are distributed round-robin across eligible
nodes, selecting lower local expert IDs first. Each node then retains that initial CUDA slot count throughout the
session; adaptive exchanges change expert identities, not capacities.

The QMoE-only `basic` strategy instead fills CUDA slots in graph topological order: all experts of earlier QMoE
nodes are on GPU until the GPU expert budget is exhausted, and later nodes are on CPU. If the boundary cuts through
a node, its lower local expert IDs stay on GPU, preserving the exact requested global CPU count. Graph scopes
are ordered lexicographically, with nodes ordered topologically within each scope. This strategy rejects sessions
containing CUDA dense `MoE` nodes when offload is enabled. Saved popularities do not affect placement, and no
adaptive exchanges are scheduled, even with a nonzero swap limit. Counters can still be collected.
Select it with `options.add_session_config_entry("session.moe_cpu_offload_strategy", "basic")`, or pass
`--strategy basic` to the benchmark runner. A zero CPU expert count leaves the ordinary all-GPU path unchanged.

The saved state uses `{alpha, epsilon, experts}`, with `{moe, expert, popularity}` records. `moe` is the zero-based
node ordinal after sorting by graph scope and node index. Loading validates bounds, uniqueness, and finite,
non-negative popularities; omitted experts start at zero. The file's alpha and epsilon override separately
configured values and are propagated to the kernels.

Pilot-initialization callbacks run after session-global placement is finalized. Registration and destruction cross
the shared-provider bridge without changing the `OpKernel` virtual table.

### Dense MoE storage and CPU computation

CPU and CUDA experts occupy disjoint compact weight stores. CPU experts use MLAS-compatible matrix layouts;
CUDA experts use the CUDA runner's layouts. The kernel retains shared initializer ownership during initialization
instead of copying the complete expert set into a CPU vector. External initializers remain file-backed mappings.

Selected CUDA experts are uploaded directly into compact storage through the external-data loader's two 64 MiB
pinned buffers. Only the CPU subset is converted to its CPU GEMM layout. Disabling that loader uses synchronous
per-expert uploads, not a full-model CPU staging copy. Inline and user-supplied initializers retain their existing
owner until initialization completes.

FP16 CPU experts use the MLAS half-precision path; BF16 experts use the dense CPU fallback. Activation, optional
biases, routing weights, and output combination preserve the operator's semantics.

### Quantized QMoE storage and CPU computation

QMoE compacts CUTLASS-packed weights, scales, derived zero-point biases, and optional explicit biases to the selected
CUDA experts. CPU execution has two paths:

| Path | Conditions and representation |
|---|---|
| GPU-packed Q4 | Available x86 MLAS Q4 FP32 backend, INT4 weights, FP16 activations, blockwise scales, and fused interleaved SwiGLU. Hidden and intermediate reduction dimensions must be divisible by the block size and by 64; the packed sizes/layouts must match the supported MLAS representation. |
| Dense fallback | Other supported cases, including INT8, BF16, per-channel scales, and other activations, dequantize CPU-resident experts into the dense MLAS representation. |

The Q4 path reverses the CUDA CUTLASS layout and produces MLAS INT4-packed weights on GPU. Initial placement and
outgoing exchanges transfer compact INT4 bytes to CPU without CPU weight prepacking or full dense dequantization.
Initial GPU packing scratch is bounded to 64 MiB, or one expert if larger. Canonical host weight bytes are released
after compact stores are ready. CPU QNBit GEMM accumulates in FP32; it may dequantize temporary tiles internally,
but does not retain full dense CPU expert matrices.

The dense fallback retains canonical quantized bytes for preparing CPU experts. Both paths retain the packed
representations needed to move a CPU-resident expert back into a CUDA slot, including its associated scales and
biases. Compact expert storage is not a bound on total process RSS or initialization peak memory.

### Hybrid execution and bounded activation buffers

CUDA routing compacts only rows selecting at least one CPU-resident expert, in chunks of at most 256 input rows.
Each node reuses one chunk's pinned-host and CUDA activation buffers rather than retaining a full-prefill
high-water allocation.

For dense MoE on devices supporting CUDA stream memory operations, and for QMoE's worker path, a persistent
per-node CPU worker first receives the four-byte row count. Nonempty batches then transfer compact activations
and routing, execute CPU expert GEMMs, and upload compact contributions while CUDA expert GEMMs proceed.
Output scattering combines both contributions. Dense MoE retains a synchronous path on devices without the
required stream memory operations.

The worker must enqueue its upload before the compute stream inserts the upload dependency; a wait on a
CUDA-invisible future producer can deadlock. Dense MoE's empty single-row CPU batch releases the completion flag
directly on GPU. With in-place dense MoE outputs, mixed CPU/CUDA execution snapshots aliased inputs before CUDA
overwrites rows needed by later CPU chunks. All-CPU execution clears output a chunk at a time after packing its input.

### Counters and asynchronous exchanges

Every node invocation decays each expert counter once:

```text
counter_next(e) = alpha * counter(e) + (1 - alpha) * contribution(e)
```

Unselected experts contribute zero. With probability weight `lambda`, a selected expert contributes
`(1 - lambda) + lambda * p(e)`, where `p(e)` is its maximum selected routing probability during the invocation.
The default `lambda=0` counts whether an expert was selected, not how many rows selected it.

Offloaded CUDA kernels update counters on GPU. The executor commits successful usage at `EndRun`, preserving
selection order and propagating asynchronous errors. A failed kernel does not commit usage; earlier successful
invocations still commit if later work fails. Only a successful overall run schedules new exchanges.

A node with experts on both devices proposes its hottest CPU expert against its coldest CUDA expert when:

```text
cpu_max > (1 + epsilon) * cuda_min
```

Deterministic expert-identity ties apply. At most one exchange is pending per node and the session enforces the
configured per-device concurrency limit. Eligible unscheduled candidates are reconsidered after later runs, not
stored in a persistent queue.

Dedicated streams prepare layouts and transfer experts through separate incoming/outgoing pinned buffers and an
extra CUDA staging slot. Within an exchange, D2H precedes H2D because the layout buffer is shared; exchanges
belonging to different nodes can overlap. Q4 eviction packs CPU weights on GPU; the dense QMoE fallback prepares
them on a separate worker. Every associated packed tensor moves with its expert.

An incomplete exchange leaves the old mapping usable without waiting. A later invocation publishes the replacement
only after transfers and CPU preparation complete. Staging retained across nodes is not bounded by the number of
simultaneously active exchanges.

Counter logs report selected expert IDs, pre-update scores, local/global ranks, and the currently published
`CPU`/`CUDA` placement. End-of-run swap records identify scheduled exchanges and concurrency-limited candidates.
See [MoE expert counting](../MoEExpertCounting.md) for the logging and state formats, and
[MoE/QMoE CUDA operators](../contrib_ops/cuda/moe_qmoe.md) for operator details.

## QMoE INT4 benchmark: adaptive strategy

This series replaces the previous FP16/INT4 sweeps, historical controls, and tuning results.
Only **Qwen3.5-35B-A3B INT4** is measured, with 40 QMoE nodes, 256 experts per node, and top-k 8.
`K` is the session-global CPU-resident expert count out of 10,240, not the number of CPU layers.

The fixed settings are alpha `0.9`, epsilon `0.1`, probability weight `0`, and at most **five in-flight swaps per
device**. Adaptive swaps are configured for every nonzero `K`, but require both CPU and CUDA expert slots.
At `K=10240`, all QMoE experts reside on CPU; the remaining model operations still use CUDA, so this is not a
CPU-only execution-provider run. `K=0` disables CPU offload and is measured both with and without CUDA Graph
capture. All nonzero `K` runs disable capture, which is unsupported with CPU offload.
No saved counter state or counter logging is used. Runtime defaults are unchanged.

### Protocol and metric definitions

Measured on 2026-10-16 using source revision `1090c0ff914a232d28b30538c5863fb925605041`, which includes the
CUDA shared-provider and aliased-input fixes. Target `main` was refreshed before measurement and was already
included in this branch. Hardware: NVIDIA H200 (SM90), driver 580.105.08, and Intel Xeon Platinum 8480C
(two sockets, 96 physical cores, no SMT). Software: Linux, Python 3.12, GenAI CUDA 0.15.2, source-built ORT 1.32.0,
and CUDA 13.3.73. Release build, native CUDA architecture, Flash/MEA enabled, quick build disabled.
GenAI loaded the rebuilt ORT from the build directory, not an installed ORT wheel.

The twelve configurations are `K=0` with capture off/on and `K=1024, 2048, ..., 10240` with capture off.
Each runs once, serially in a fresh process on physical H200 GPU 0. The intermediate `K=1024..8192` measurements
are retained from the initial sweep; the two all-GPU controls and `K=9216, 10240` were measured in an extension
using the same model, prompts, settings, and unchanged runtime/provider binaries.
Every run uses the unchanged [five prompts](benchmark-prompts.json), greedy generation, and exactly 256 generated
tokens per prompt. The first two prompts are warmup.

- **Prefill time (ms/prompt):** arithmetic mean of `append_tokens()` wall times for prompts 3-5.
- **Decoded tokens/s:** 762 steady-state decoded tokens divided by their total measured decode time across
  prompts 3-5. The generation call using prefill logits and the first actual decode inference of each prompt
  are excluded. Prefill, loading, and Python log-writing outside the timed calls are not included.
- **Peak GPU (GiB):** maximum device-wide NVML usage sampled every 50 ms across loading and all five prompts,
  including ambient GPU memory, not just model allocations.
- **Peak CPU RSS (GiB):** maximum process RSS across loading and all five prompts, taking the larger of the
  50 ms samples and Linux `ru_maxrss`. This is resident memory, not total allocated or virtual memory.

Single runs are not repeated statistical estimates. Greedy generation does not replay an identical routing trace,
and this benchmark does not evaluate model quality.

A separate Nsight Systems smoke run (one prompt, 16 generated tokens, `K=0`, `--cuda-graph`) confirmed actual
capture and replay: one `cudaStreamBeginCapture`, one `cudaStreamEndCapture`, one `cudaGraphInstantiate`,
and 15 `cudaGraphLaunch` calls. Its instrumented timings are excluded from the table.

The model graph SHA-256 is `84fe3652ac0c711b8f7782a93459296a39f58abc654d75dfb42d54ef474f2b03`.
Runtime and CUDA provider hashes identify the exact binaries:

| Artifact | SHA-256 |
|---|---|
| `libonnxruntime.so.1.32.0` | `5750dc0a84f95fb92b2d588fce1a50194671bd73aa7808b50659377c9879abf7` |
| `libonnxruntime_providers_cuda.so` | `8e7e4fb2fb4fe488fb1b603b1d11c9ac573c74b94e6a3db27f6e95f9b7ada74b` |

### Results

All twelve runs completed successfully. The runtime/provider hashes were unchanged after the series.
Each row is one run, with warm timing over prompts 3-5 and memory peaks over the complete process.
Raw per-prompt JSON and logs are retained locally, not committed.

| K: CPU experts | CUDA Graph | Prefill time (ms/prompt) | Decoded tokens/s | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 21.077 | 129.257 | 39.390 | 1.574 |
| 0 | On | 21.784 | 168.870 | 39.510 | 1.646 |
| 1024 | Off | 34.407 | 79.497 | 38.224 | 18.263 |
| 2048 | Off | 43.612 | 67.150 | 36.539 | 18.387 |
| 3072 | Off | 140.587 | 40.814 | 34.732 | 18.472 |
| 4096 | Off | 226.890 | 36.228 | 32.996 | 18.612 |
| 5120 | Off | 246.045 | 35.727 | 31.160 | 21.733 |
| 6144 | Off | 256.879 | 32.769 | 29.615 | 25.308 |
| 7168 | Off | 366.267 | 28.065 | 27.930 | 28.883 |
| 8192 | Off | 439.508 | 26.730 | 26.722 | 32.460 |
| 9216 | Off | 469.877 | 25.334 | 24.601 | 36.035 |
| 10240 | Off | 744.964 | 26.572 | 23.163 | 39.592 |

## QMoE INT4 benchmark: basic strategy

This series uses `--strategy basic` with the same Qwen3.5 INT4 model, unchanged five prompts, 256 greedy tokens
per prompt, first two prompts as warmup, and the four metric definitions above. Alpha is `0.9`, epsilon is `0.1`,
probability weight is `0`, and the configured swap limit remains `5`. The strategy prevents all swaps regardless
of that limit. The sweep covers `K=0, 1024, ..., 10240`, plus a separate captured all-GPU control at `K=0`.
With 256 experts per node, every nonzero step moves exactly four trailing QMoE nodes to CPU; there is no split
boundary node in this model. All-GPU controls disable offload and its instrumentation entirely.

Measured on 2026-10-16 using source revision `487a44eddac3a81fd0ba63757e5aa8fdf9a2f337`, including the basic
strategy and failed-run logger-lifetime fix. Target `main` was refreshed and was already included. The software,
host, and build configuration match those described above. These runs use physical H200 **GPU 1**, because GPU 0
was running another workload. This is a shared host, not a fully isolated benchmark machine. The earlier adaptive
series used GPU 0 and different binaries, so differences between the two tables are not a controlled,
same-binary comparison of strategies.

| Artifact | SHA-256 |
|---|---|
| `libonnxruntime.so.1.32.0` | `c0293a3f964737f4867a1826443c4a04a2128d5f071055243f0563ef956f5116` |
| `libonnxruntime_providers_cuda.so` | `cd0bfd93995866a47dae2a98a99a0c9df39d007875e677efc4c1168ce17bed25` |

A separate untimed real-model check at `K=4096` confirmed that the first 24 QMoE nodes selected only CUDA-resident
experts and the last 16 selected only CPU-resident experts over 16 invocations per node, with no swap records.
Its logging-enabled timings are not included in the results.

A separate Nsight Systems run of the new all-GPU control (one prompt, 16 generated tokens) confirmed one capture
and instantiation and 15 `cudaGraphLaunch` calls. Its instrumented timings are also excluded.
All twelve timed runs completed successfully, with unchanged runtime/provider hashes. Each row is a single run;
raw per-prompt JSON and logs are retained locally, not committed.

| K: CPU experts | CUDA Graph | Prefill time (ms/prompt) | Decoded tokens/s | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 21.079 | 128.563 | 39.388 | 1.574 |
| 0 | On | 21.807 | 169.488 | 39.447 | 1.645 |
| 1024 | Off | 73.010 | 89.572 | 37.930 | 18.144 |
| 2048 | Off | 122.929 | 69.399 | 36.275 | 18.144 |
| 3072 | Off | 187.570 | 57.557 | 34.543 | 18.144 |
| 4096 | Off | 217.239 | 50.454 | 32.851 | 18.152 |
| 5120 | Off | 209.857 | 45.857 | 31.127 | 21.726 |
| 6144 | Off | 329.509 | 40.231 | 29.431 | 25.302 |
| 7168 | Off | 418.698 | 36.668 | 27.771 | 28.781 |
| 8192 | Off | 463.234 | 35.093 | 26.047 | 32.356 |
| 9216 | Off | 572.767 | 28.639 | 24.351 | 35.915 |
| 10240 | Off | 611.970 | 28.730 | 22.691 | 39.594 |

## Reproducing the series

Use the Qwen3.5 INT4 GenAI export and a matching source-built ORT/CUDA provider. With the build output on
`PYTHONPATH`, run from outside the source package directory so Python imports the compiled package:

```bash
export PYTHONPATH=/path/to/onnxruntime/build/linux_cuda/Release
export LD_LIBRARY_PATH=/path/to/onnxruntime/build/linux_cuda/Release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd /path/to/onnxruntime/build/linux_cuda/Release
strategy=adaptive  # Set to basic for fixed GPU-first QMoE placement.
gpu=0             # Use a free physical H200 GPU; the basic series used GPU 1.
output="/tmp/qwen35-int4-${strategy}-sweep"
mkdir -p "$output"
for k in 0 1024 2048 3072 4096 5120 6144 7168 8192 9216 10240; do
  CUDA_VISIBLE_DEVICES="$gpu" python /path/to/onnxruntime/tools/python/benchmark_moe_cpu_offload.py \
    --name "qwen35-int4-${strategy}-k${k}-alpha09-epsilon01-swaps5" \
    --model /path/to/qwen3.5-35b-cuda-int4 \
    --prompts /path/to/onnxruntime/docs/moe-cpu-offload/benchmark-prompts.json \
    --max-new-tokens 256 \
    --cpu-offload-experts "$k" \
    --strategy "$strategy" \
    --counter-alpha 0.9 \
    --swap-epsilon 0.1 \
    --max-in-flight-swaps-per-device 5 \
    --swap --no-counter-log --no-cuda-graph \
    --output "$output/k${k}.json" \
    --log "$output/k${k}.log" || exit "$?"
done

CUDA_VISIBLE_DEVICES="$gpu" python /path/to/onnxruntime/tools/python/benchmark_moe_cpu_offload.py \
  --name "qwen35-int4-${strategy}-k0-cuda-graph" \
  --model /path/to/qwen3.5-35b-cuda-int4 \
  --prompts /path/to/onnxruntime/docs/moe-cpu-offload/benchmark-prompts.json \
  --max-new-tokens 256 \
  --cpu-offload-experts 0 \
  --strategy "$strategy" \
  --counter-alpha 0.9 \
  --swap-epsilon 0.1 \
  --max-in-flight-swaps-per-device 5 \
  --swap --no-counter-log --cuda-graph \
  --output "$output/k0-graph.json" \
  --log "$output/k0-graph.log"
```

Use `--no-swap` to freeze adaptive placement, `--state` to load initial popularities, `--popularity-output` to save them,
and `--counter-log` for separate routing analysis. Those runs are not part of this fixed-parameter series.
