# CPU offload and adaptive expert placement for MoE and QMoE

The built-in CUDA execution provider can execute selected experts on CPU while keeping the remaining experts on
CUDA. Both static placement and asynchronous, counter-driven expert exchanges are implemented for
`com.microsoft::MoE` and `com.microsoft::QMoE`. The graph nodes remain assigned to CUDA: CPU offload is internal to
their kernels, not execution-provider fallback or a graph rewrite.

The implementation and measurements below accompany [PR #33347](https://github.com/microsoft/onnxruntime/pull/33347).
The measured model is **Qwen3.5-35B-A3B**. Historical measurements are labeled with their measured
revision; they are not new measurements of the latest source.

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
| `session.moe_expert_counter_alpha` | `0.75` | Finite counter decay coefficient in `[0, 1]`. |
| `session.moe_expert_counter_probability_weight` | `0` | Finite blend in `[0, 1]` between binary selection and routing probability. |
| `session.moe_expert_swap_epsilon` | `0.01` | Finite, non-negative relative margin required for an exchange. |
| `session.moe_max_in_flight_swaps_per_device` | `4` | Non-negative exchange limit per CUDA device; `0` keeps placement static. |
| `session.moe_expert_counter_state_file` | Unset | Input JSON containing `alpha`, `epsilon`, and per-expert popularities. |
| `session.moe_expert_popularity_file` | Unset | Output JSON written when the session is destroyed. |

For example, configure an ordinary CUDA session with 1,000 CPU experts and adaptive exchanges:

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_session_config_entry("session.moe_cpu_offload_experts", "1000")
options.add_session_config_entry("session.moe_expert_counter_alpha", "0.9")
options.add_session_config_entry("session.moe_expert_swap_epsilon", "0.1")
options.add_session_config_entry("session.moe_max_in_flight_swaps_per_device", "4")

session = ort.InferenceSession("model.onnx", options, providers=["CUDAExecutionProvider"])
```

The example's alpha and epsilon are benchmark settings, not runtime defaults.

## Current implementation

### Initial placement and ownership

The session owns the global counters, initial placement policy, and a `KernelPilot` for each registered kernel.
After registration and optional state loading, it assigns:

```text
CUDA expert count = eligible expert count - configured CPU expert count
```

With nonzero saved counters, the highest-ranked eligible experts receive CUDA slots. Ties use local expert ID,
then graph scope and node index. If all scores are zero, CUDA slots are distributed round-robin across eligible
nodes, selecting lower local expert IDs first. Each node then retains that initial CUDA slot count throughout the
session; adaptive exchanges change expert identities, not capacities.

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

## Benchmark results from PR #33347

These are recorded measurements, not a new benchmark of the latest merge or aliasing fix. All latency values below
measure real model execution, not counter-trace replay. Greedy generation does not replay identical routing, and
these runs are not a model-quality evaluation.

### Environment and protocols

The host was an Intel Xeon Platinum 8480C system with 96 physical cores across two sockets/NUMA nodes, without SMT.
The GPU was an NVIDIA H200, SM90, with 143,771 MiB; the recorded driver was 580.105.08. The FP16 environment used
Linux, Python 3.12, GenAI 0.15.2, ORT 1.32.0, ONNX 1.23.2, and CUDA 13.3.73, with a Release native-architecture build,
Flash/MEA enabled, and quick build disabled.

FP16 measurements used physical GPU 5 and revision `63148bd3b96a7c38a15b7b8f265df44b7f0e3a54`, before the
bounded-initialization update `54bdf402e0`. GPU-packed INT4 measurements used physical GPU 0 and revision
`bbbf10078b`. Historical controls are identified separately; the two model formats are not matched before/after
measurements.

Unless stated otherwise, runs used the committed [five prompts](benchmark-prompts.json), 256 greedily generated
tokens per prompt, fresh processes executed serially, adaptive exchanges enabled, a four-exchange limit,
probability weight zero, counter logging disabled, and CUDA Graph disabled. Prefill was enabled but measured
separately from decode. Runtime/provider hashes remained unchanged within each sweep.

FP16 scores average 765 actual decode inferences from prompts 3-5, excluding the first two prompts, prefill, and
the generation call using prefill logits. INT4 steady-decode scores additionally exclude each retained prompt's
first actual decode inference, leaving 762 inferences. Memory peaks include initialization and all prompts;
GPU memory is device-wide NVML usage. A single run is not a repeated statistical estimate.

### FP16 MoE: alpha sweep

Epsilon `0.1`, 1,000 CPU experts. SD is sample standard deviation; `n/a` means only one run.

| Alpha | Runs | Decode mean +/- SD (ms/token) | Run range | GPU after load (GiB) | Peak GPU (GiB) |
|---|---:|---:|---|---:|---:|
| 0.5 | 1 | 19.270 +/- n/a | 19.270-19.270 | 61.03 | 62.91 |
| 0.75 | 1 | 22.099 +/- n/a | 22.099-22.099 | 60.78 | 62.67 |
| 0.9 | 2 | 21.557 +/- 0.706 | 21.058-22.056 | 60.78 | 62.67 |
| 0.95 | 2 | 21.318 +/- 0.862 | 20.709-21.927 | 60.66 | 62.55 |
| 0.99 | 1 | 20.607 +/- n/a | 20.607-20.607 | 61.03 | 62.91 |

### FP16 MoE: epsilon sweep

Alpha `0.9`, 1,000 CPU experts.

| Epsilon | Runs | Decode mean +/- SD (ms/token) | Run range | GPU after load (GiB) | Peak GPU (GiB) |
|---|---:|---:|---|---:|---:|
| 0 | 2 | 21.552 +/- 1.451 | 20.526-22.578 | 60.54 | 62.43 |
| 0.01 | 1 | 22.544 +/- n/a | 22.544-22.544 | 61.03 | 62.91 |
| 0.05 | 1 | 22.847 +/- n/a | 22.847-22.847 | 60.53 | 62.42 |
| 0.1 | 2 | 21.557 +/- 0.706 | 21.058-22.056 | 60.78 | 62.67 |
| 0.2 | 1 | 21.468 +/- n/a | 21.468-21.468 | 60.53 | 62.42 |
| 0.5 | 2 | 20.233 +/- 0.523 | 19.863-20.602 | 60.65 | 62.54 |

### FP16 MoE: CPU expert-count sweep

Alpha `0.9`, epsilon `0.1`.

| CPU experts | Runs | Decode mean +/- SD (ms/token) | Run range | GPU after load (GiB) | Peak GPU (GiB) |
|---|---:|---:|---|---:|---:|
| 0 | 2 | 9.059 +/- 1.127 | 8.263-9.856 | 66.31 | 68.56 |
| 1 | 2 | 9.370 +/- 0.141 | 9.270-9.469 | 66.67 | 68.94 |
| 100 | 2 | 11.558 +/- 0.726 | 11.044-12.071 | 66.05 | 67.94 |
| 256 | 2 | 13.781 +/- 1.045 | 13.042-14.520 | 65.13 | 67.03 |
| 512 | 1 | 14.770 +/- n/a | 14.770-14.770 | 63.38 | 65.28 |
| 768 | 1 | 19.895 +/- n/a | 19.895-19.895 | 62.39 | 64.28 |
| 1024 | 2 | 20.870 +/- 0.727 | 20.355-21.384 | 60.51 | 62.40 |
| 1280 | 2 | 26.078 +/- 1.672 | 24.896-27.261 | 59.01 | 60.90 |
| 1536 | 1 | 28.509 +/- n/a | 28.509-28.509 | 57.64 | 59.53 |
| 1792 | 2 | 44.844 +/- 2.725 | 42.917-46.770 | 56.02 | 57.91 |
| 2048 | 2 | 84.472 +/- 5.616 | 80.502-88.443 | 54.51 | 56.26 |

The FP16 sweeps contain 33 completed runs across 21 distinct configurations; alpha/epsilon share their reference
runs. This is not a completed three-repetition sweep. Memory entries are medians across available runs.
Multiples of 256 are whole-layer-equivalent counts, not necessarily entire CPU-resident layers.

At 1,000 CPU experts the reference is 21.557 +/- 0.706 ms/token. Alpha `0.5` gives the lowest single observation,
19.270 ms/token; epsilon `0.5` averages 20.233 +/- 0.523 ms/token. These do not demonstrate a stable 10 ms/token
improvement. Larger CPU counts save GPU memory but sharply increase decode latency. All-GPU zero is a baseline,
not an adaptive-offload optimization.

### INT4 QMoE: full GPU, dynamic offload, and full CPU

Single runs used three prompts and 128 generated tokens per prompt, reporting warm prompts 2-3, excluding model
load, prefill, and the first decode call. Dynamic placement used 1,000 CPU experts, alpha `0.9`, epsilon `0.1`,
and exchanges enabled. Full GPU and full CPU used no offload or exchanges.

| Placement | Warm mean (ms/token) | Mean prompt median (ms/token) | Warm max (ms/token) | Load (s) | Peak CPU RSS (MiB) | Peak GPU (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Full GPU, CUDA | 7.784 | 7.758 | 9.506 | 4.541 | 1,606.8 | 40,277.8 |
| Previous dynamic offload, dense CPU experts | 53.425 | 51.872 | 115.213 | 41.907 | 27,157.7 | 38,945.8 |
| GPU-packed Q4 dynamic offload | 14.208 | 13.884 | 110.326 | 19.877 | 18,703.5 | 39,053.8 |
| Full CPU provider | 3,242.951 | 3,233.941 | 3,454.335 | 13.651 | 23,429.8 | 614.9 |

The CPU-only GPU value is ambient usage, not model allocation. Full GPU/full CPU and the dense-offload row are
historical controls; the GPU-packed row uses `bbbf10078b` on physical GPU 0. The observed Q4 speedup over the old
dense CPU path is 3.76x, with lower load time and CPU RSS. These are single runs, not repeated averages or
identical-routing replay. The new run's three prefill times were 1,302.2, 44.3, and 25.4 ms.

For the five-prompt, 256-token protocol, GPU-packed Q4 averaged 12.434 ms/token versus 26.294 ms/token for the
previous dense CPU path, both with 1,000 CPU experts, alpha `0.9`, epsilon `0.1`, and exchanges enabled.

A separate Nsight capture observed 4,402 `PackMlasQ4WeightsKernel` launches totaling 31.55 ms of GPU execution,
CPU samples in `SQ4BitGemmM1Kernel_CompFp32`, and no dense FP16 GEMM leaf samples. Instrumented timings are
excluded from latency comparisons.

### GPU-packed INT4 QMoE: CPU expert-count sweep

Revision `bbbf10078b`, alpha `0.9`, epsilon `0.1`; one run per configuration. Warm prefill averages prompts 3-5.
The measured count really is 3062, not 3072.

| CPU experts | Warm decode (ms/token) | Warm prefill (ms/prompt) | Peak GPU (GiB) | Peak CPU RSS (GiB) | Load (s) |
|---|---:|---:|---:|---:|---:|
| 1024 | 12.944 | 37.7 | 38.22 | 18.26 | 16.40 |
| 2048 | 16.792 | 59.8 | 36.56 | 18.39 | 19.88 |
| 3062 | 23.627 | 185.2 | 34.77 | 18.48 | 20.78 |
| 4096 | 26.205 | 195.9 | 32.95 | 18.61 | 22.44 |
| 5120 | 29.944 | 266.9 | 31.15 | 21.73 | 25.03 |
| 6144 | 31.244 | 308.1 | 29.61 | 25.31 | 27.11 |

Moving from 1024 to 6144 CPU experts saved 8.61 GiB of peak GPU usage, while decode rose from 12.944 to
31.244 ms/token.

### GPU-packed INT4 QMoE: routing at 4096 CPU experts

A separate counter-logged run used `bbbf10078b`, 4096 CPU experts out of 10240, alpha `0.9`, epsilon `0.1`,
and adaptive exchanges. Each decode invokes 40 QMoE nodes selecting eight experts each. Counts below are
expert-token executions, including repeated use, not resident or distinct expert counts.

| Scope | Decode inferences | QMoE invocations | CPU executions | GPU executions | CPU share | Mean CPU experts per QMoE |
|---|---:|---:|---:|---:|---:|---:|
| All five prompts | 1275 | 51000 | 149307 | 258693 | 36.595% | 2.928 / 8 |
| Warm prompts 3-5 | 762 | 30480 | 88173 | 155667 | 36.160% | 2.893 / 8 |

| CPU experts selected out of 8 | All-decode invocations | All-decode share | Warm invocations | Warm share |
|---:|---:|---:|---:|---:|
| 0 | 2491 | 4.884% | 1570 | 5.151% |
| 1 | 6187 | 12.131% | 3825 | 12.549% |
| 2 | 11219 | 21.998% | 6790 | 22.277% |
| 3 | 13215 | 25.912% | 7871 | 25.823% |
| 4 | 10654 | 20.890% | 6255 | 20.522% |
| 5 | 5272 | 10.337% | 3054 | 10.020% |
| 6 | 1685 | 3.304% | 951 | 3.120% |
| 7 | 265 | 0.520% | 153 | 0.502% |
| 8 | 12 | 0.024% | 11 | 0.036% |

At least one CPU expert executed in 95.116% of all decode QMoE invocations, or 94.849% in warm scope.
Mean CPU experts per invocation ranged from 1.277 at layer 3 to 3.944 at layer 21. This is execution frequency,
not CPU critical-path time.

Across all decode, 4208 distinct experts executed on CPU and 5699 on GPU; 1075 appeared on both, giving 8832
unique experts overall. CPU distinct identities can exceed the 4096 resident slots because exchanges change
placement. Prefill's 200 records contained 5805 CPU and 9329 GPU distinct selections summed per invocation;
these deduplicated records do not expose exact prefill expert-token multiplicities and are not added to decode totals.

### GPU-packed INT4 QMoE: alpha/epsilon tuning at 4096 CPU experts

Fourteen exploratory single runs used the same `bbbf10078b` build and five-prompt, 256-token protocol. The best
exploratory pair, alpha `0.75` and epsilon `0.5`, did not improve latency in repeated confirmation.

| Configuration | Alpha | Epsilon | Runs | Warm decode mean +/- sample SD (ms/token) | Individual run scores | Warm prefill (ms/prompt) |
|---|---:|---:|---:|---:|---|---:|
| Reference | 0.9 | 0.1 | 3 | 25.752 +/- 0.311 | 25.394, 25.964, 25.896 | 210.9 |
| Exploratory candidate | 0.75 | 0.5 | 3 | 26.720 +/- 1.014 | 27.128, 25.565, 27.465 | 229.9 |

Confirmation ran in interleaved order: reference, candidate, candidate, reference, reference, candidate.
It excludes exploratory measurements to avoid selection bias. The candidate was 3.76% slower on average;
no stable latency improvement was demonstrated and runtime defaults were not changed.

The exploratory reference was 29.250 ms/token, versus 26.205 ms/token in the earlier CPU-count sweep.
The following single-run search results are not confirmed optimization gains:

| Alpha | Epsilon | Warm decode (ms/token) | Warm prefill (ms/prompt) |
|---:|---:|---:|---:|
| 0.9 | 0.1 | 29.250 | 240.0 |
| 0.5 | 0.1 | 26.924 | 200.4 |
| 0.75 | 0.1 | 25.640 | 213.8 |
| 0.95 | 0.1 | 26.060 | 196.6 |
| 0.99 | 0.1 | 27.537 | 219.5 |
| 0.75 | 0 | 27.083 | 217.1 |
| 0.75 | 0.01 | 28.953 | 234.6 |
| 0.75 | 0.05 | 26.605 | 254.9 |
| 0.75 | 0.2 | 27.654 | 209.3 |
| 0.75 | 0.5 | 25.474 | 197.9 |
| 0.75 | 1.0 | 26.584 | 218.6 |
| 0.5 | 0.5 | 26.346 | 247.3 |
| 0.9 | 0.5 | 26.876 | 245.6 |
| 0.95 | 0.5 | 26.155 | 231.8 |

All 20 tuning invocations completed successfully. No saved counter state or Python replay optimizer was used.
Raw JSON results and logs were retained locally, not committed.

The independent-direction layout-buffer experiment was removed: two runs averaged 22.157 ms/token against
20.524 ms/token for bracketed controls, about 7.95% slower on predecessor source. It is not part of the current
implementation, and its isolated 18.208 ms/token observation is not a stable performance result.

## Reproducing a benchmark

Use a GenAI export containing supported MoE/QMoE nodes and a matching source-built ORT/CUDA provider.
Run from the repository root, selecting one otherwise idle physical GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/python/benchmark_moe_cpu_offload.py \
  --name qwen35-int4-cpu4096 \
  --model /path/to/qwen3.5-35b-a3b-int4 \
  --prompts docs/moe-cpu-offload/benchmark-prompts.json \
  --max-new-tokens 256 \
  --cpu-offload-experts 4096 \
  --counter-alpha 0.9 \
  --swap-epsilon 0.1 \
  --swap --no-counter-log --no-cuda-graph \
  --output /tmp/qwen35-int4-cpu4096.json \
  --log /tmp/qwen35-int4-cpu4096.log
```

Use `--no-swap` for static placement, `--state` to load initial popularities, `--popularity-output` to save them,
and `--counter-log` for routing analysis. The harness defaults to alpha `0.9` and epsilon `0.1`, unlike runtime
defaults. Keep instrumented runs separate from latency measurements. Preserve the exact source revision, model,
prompt set, provider hashes, warmup exclusions, and repetition count when comparing results.
