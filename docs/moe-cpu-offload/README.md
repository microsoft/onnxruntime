# Adaptive CUDA expert offloading for Qwen 3.6 MoE

**Status:** Implementation in progress

**Date:** 2026-09

## Objective

Implement adaptive expert placement for Qwen 3.6 and other Mixture-of-Experts (MoE) models whose expert weights do not
all fit in GPU memory.

Each `MoE` and `QMoE` node can maintain one exponentially decayed counter per expert. Adaptive offloading supports
built-in CUDA FP16/BF16 `MoE` and uniform INT4/INT8 `QMoE` with canonical
`weights_prepacked=0` initializers. Each kernel cache holds disjoint, configurable CPU and CUDA expert subsets.
FP4-family, FP8, INT2, mixed-width, and externally prepacked QMoE weights remain outside the implemented scope.

After a complete model inference, the pilot exchanges hot CPU experts with cold CUDA experts within each node when
their counter differences exceed a configurable threshold. Each node retains its configured CUDA slot count; global
CUDA slot redistribution across nodes remains future work.

Training, router changes, expert-weight quantization, and multiple CUDA devices are outside this implementation.

## Implemented foundations

- [#32261](https://github.com/microsoft/onnxruntime/pull/32261) added opt-in MoE/QMoE routing instrumentation,
  reproducible prompt and analysis tools, and the first routing measurements.
- [#32738](https://github.com/microsoft/onnxruntime/pull/32738) added the session-global expert state, CPU and CUDA
  expert-selection collection, and exponentially decayed counters. It does not implement placement, CPU offload,
  swaps, or redistribution.
- [#33039](https://github.com/microsoft/onnxruntime/pull/33039) added static FP16/BF16 CUDA `MoE` placement, hybrid
  CPU/CUDA execution, and compact CUDA expert storage.
- Dense MoE exchanges hot CPU experts with cold CUDA experts asynchronously while preserving each node's CUDA slot
  count. Global slot redistribution across nodes remains future work.
- CUDA INT4/INT8 `QMoE` compacts the CUTLASS-packed weights, scales, derived zero-point biases, and optional explicit
  biases to the selected CUDA experts. Blockwise INT4/FP16 fused SwiGLU on x86 uses GPU-produced MLAS Q4 packed weights
  and CPU QNBit GEMM with FP32 accumulation; other supported cases retain the dense MLAS fallback. QMoE copies only
  the compact row count when no CPU route is selected; non-empty CPU work runs on a persistent worker and overlaps
  CUDA expert kernels. QMoE swaps move every packed expert tensor together on dedicated streams. The outgoing expert
  is packed directly from the CUDA slot into CPU INT4 layout on the D2H stream for Q4, or dequantized on a separate
  worker for the dense fallback. The new map is published only after transfers and CPU preparation complete.
- Expert swaps transpose matrix weights between CUDA and MLAS layouts on their dedicated CUDA streams, rather than
  blocking the transfer pipeline with a CPU transposition callback. Incoming and outgoing experts use separate pinned
  buffers; publication waits for both transfers to finish before changing the CPU/CUDA maps.

## Exploratory routing analysis

An exploratory trace was collected from Qwen3.5-35B-A3B INT4 on CUDA using 10 prompts. It contains 59,880 valid routing
records from 40 QMoE layers, with 256 experts per layer and top-k 8. The frequency ranks below are zero-based and
learned from the complete trace. These preliminary results demonstrate the routing concentration but are not an
end-to-end offload performance measurement.

The first graph compares the fraction of routing decisions selecting an expert at or above a frequency-rank threshold
with the fraction of expert-weight bytes that can be offloaded at that threshold. Keeping only the 64 most frequent
experts of each layer on CUDA retains about 25.1% of expert bytes and covers about 68.9% of observed selections. Keeping
the first 128 experts retains about 50.2% of expert bytes and covers about 90.3% of selections. Keeping the first 192
retains about 75.3% of expert bytes and covers about 98.4% of selections. The curve therefore shows substantial routing
concentration, but it also shows that an aggressive memory reduction sends a meaningful fraction of expert work to CPU.

[![Normalized routing coverage and expert bytes](images/normalized-total-vs-expert-bytes.png)](images/normalized-total-vs-expert-bytes.png)

The second graph compares the same rank-threshold distribution for representative early, middle, and late QMoE layers.
The shapes differ substantially: for example, layer 20 concentrates its selections among fewer experts, while layer 0
uses a broader portion of its expert set. A single fixed per-layer CUDA capacity is therefore unlikely to use the global
budget efficiently. This motivates per-node counters and the second implementation step's redistribution across nodes.

[![Selected QMoE layer expert-rank distributions](images/selected-layers-expert-ranks.png)](images/selected-layers-expert-ranks.png)

## Global offload configuration

The numerical policy parameters are exposed as session configuration entries:

| Session option | Parameter | Meaning and valid range |
|---|---|---|
| `session.moe_cpu_offload_experts` | Offload count | Global number of experts to execute from CPU (`>= 0`); default `0`. |
| `session.moe_expert_counter_alpha` | `alpha` | Counter decay coefficient in `[0, 1]`; selected experts receive `1 - alpha`; default `0.75`. |
| `session.moe_expert_counter_probability_weight` | Probability weight | Blend in `[0, 1]` between binary selection and routing probability; default `0`. |
| `session.moe_expert_swap_epsilon` | `epsilon` | Relative swap margin, finite and `>= 0`; default `0.01`. |
| `session.moe_max_in_flight_swaps_per_device` | Swap concurrency | Maximum concurrent adaptive swaps per CUDA device (`>= 0`); `0` disables adaptive swaps; default `4`. |

The optional `session.moe_expert_counter_state_file` path is configured separately.

The CPU offload count has the following meaning:

- A non-negative integer is the total number of experts to offload to CPU.
- Negative values, non-integer values, and counts larger than the total number of eligible CUDA FP16/BF16 `MoE` plus
  supported INT4/INT8 `QMoE` experts are invalid.
- Zero disables expert offloading.

The global CUDA budget is:

```text
cuda_expert_count = total_expert_count - cpu_offload_expert_count
```

One CUDA slot contains all weights required to execute one expert. The kernel cache splits weights into disjoint
expert subsets: CPU-resident experts use compact MLAS-layout storage, while CUDA-resident experts use compact CUDA
storage. Dense MoE does not retain a canonical CPU copy of every CUDA expert. The QMoE Q4 path reverses the SM80
CUTLASS layout and packs MLAS INT4 weights on GPU, then releases the canonical host weight copies. Initial packing
uses at most 64 MiB of GPU scratch (or one expert if larger); swaps use one expert's scratch. The dense QMoE fallback
retains canonical quantized CPU bytes for dequantization. Both paths keep CUTLASS-packed host bytes for CPU-resident
experts so an incoming expert can replace a CUDA slot directly. An exchange
preserves both stores' capacities and publishes the mapping only when the replacement slot is ready. This describes
the live expert cache, not a bound on total process RSS or temporary initialization/prepacking storage.

## Expert counters

Each `MoE` and `QMoE` node owns one counter for each of its experts. After the node executes, every counter is updated
once:

```text
count_{t+1}(e) = alpha * count_t(e) + (1 - alpha) * (1 if expert e was used, otherwise 0)
```

The used term is binary for one invocation. Selecting the same expert for several rows still contributes `1`, not the
number of routed rows.

For CPU-offloaded CUDA MoE nodes, `session.moe_expert_counter_probability_weight=lambda` changes the selected-expert
contribution to `(1 - lambda) + lambda * probability`. Repeated selections within one invocation use the maximum
routing probability observed for that expert. `0` preserves binary counting and `1` uses only routing probability.

CPU-offloaded CUDA nodes update their counters on the GPU. The pilot consumes counter snapshots and selected-expert
metadata at the end of the run, before scheduling swaps, rather than waiting for routing on the host at every node.
Only successful kernel invocations are committed, and asynchronous CPU execution errors are propagated by `EndRun`.
Repeated invocations of the same node consume the preceding snapshot before reusing its buffers.

For dense MoE on devices supporting CUDA stream memory operations, routing compacts only rows that select a
CPU-resident expert.
A short notification callback wakes a per-node CPU worker only for a nonempty compacted batch. That worker transfers
the compacted activations and routing, executes MLAS, and uploads only those rows' contributions. CUDA expert kernels
can run concurrently; the compute stream waits for the CPU result only before scattering and combining it.
At that boundary, the host must ensure the worker has enqueued its upload before inserting the CUDA upload event and
stream-memory wait. A wait on a future, CUDA-invisible upload alone can deadlock during subgraph execution or tactic
profiling. The upload dependency makes the producer visible to CUDA without moving the wait ahead of CUDA expert GEMMs.
Single-row decode with no CPU routes releases that device-side wait directly on GPU, without waiting for the host
notification. Devices without stream memory operations retain the synchronous CPU execution path.

INT4/INT8 QMoE uses the same bounded row compaction and scatter kernels, but currently completes each compact CPU
chunk synchronously before launching that tile's CUDA expert work. This preserves bounded memory and correctness for
mixed and all-CPU placement; overlapping QMoE CPU execution with CUDA expert GEMMs is future optimization work.

The policy exposes `alpha` in `[0, 1]` and non-negative `epsilon`. Counters are updated in place and remain bounded by
the larger of their initial value and `1`. Their defaults and constraints are covered by option-parsing tests.

Expert identity is `(graph_scope, node_index, node_type, expert_id)`. Ranking is by descending counter. Ties are resolved
by `expert_id`, then graph scope and node index, so placement is deterministic.

When counter logging is enabled, each event reports only the experts selected by that invocation. Each selected-expert
record contains the local `expert_id`, its pre-update `score`, its zero-based `selected_rank` among the selected
experts, its zero-based `node_rank` among every expert in the node, its zero-based `global_position` among every
expert in the model, and its current `device` (`CPU` or `CUDA`). Ranks use descending score. Local ties are broken by
expert ID, while global ties are broken by the fixed model-wide expert index. The device reflects the currently
published placement; an asynchronous swap is not visible until the kernel publishes it. Each event also reports
`moe_count` and `total_expert_count`.

At the end of each successful run, `moe_expert_swaps` reports the number of in-flight swaps and eligible candidates
blocked by the concurrency limit, both per CUDA device and in total. Candidates are recalculated after every run and
are not retained in a persistent queue. `scheduled_swaps` identifies every exchange started by that run, including
the node, the outgoing CUDA expert, the incoming CPU expert, and their counter scores.

```json
"selected_experts": [
  {
    "expert_id": 7,
    "score": 0.1,
    "selected_rank": 0,
    "node_rank": 3,
    "global_position": 1024,
    "device": "CUDA"
  }
]
```

```json
"scheduled_swaps": [
  {
    "graph_scope": "main",
    "node_index": 12,
    "device_id": 0,
    "outgoing_cuda_expert": 41,
    "incoming_cpu_expert": 173,
    "outgoing_score": 0.05,
    "incoming_score": 0.2
  }
]
```

## Initial counter state

An optional session setting names a UTF-8 JSON file containing initial counter values and parameters:

```text
session.moe_expert_counter_state_file=<path-to-popularity.json>
```

The file uses the same `{alpha, epsilon, experts}` format produced by `session.moe_expert_popularity_file`. Loading
restores both parameters and each `{moe, expert, popularity}` record. JSON values replace explicit alpha and epsilon
session settings. Loading validates MoE and expert bounds, record uniqueness, and finite non-negative popularities.
Experts omitted from the file start at zero.

Set `session.moe_expert_popularity_file=<path>` to write the final popularity counters as JSON when the session is
destroyed. The file includes `alpha`, `epsilon`, and one `{moe, expert, popularity}` record per expert. `moe` is the
stable zero-based ordinal obtained by sorting resolved nodes by graph scope and node index. The benchmark wrapper
exposes the same setting as `--popularity-output`.

The first offload implementation uses loaded counters to select the highest-ranked experts globally. If all counters
are zero, it distributes CUDA slots round-robin across eligible nodes and selects lower local expert IDs first. This
placement remains fixed when adaptive swaps are disabled. With swaps enabled, each node retains its initial CUDA
slot count, but the experts assigned to those slots can change after successful inference boundaries.

## Per-node placement update

One invocation uses an immutable expert-to-device mapping:

- CUDA-resident experts execute on CUDA.
- CPU-resident experts execute on CPU.
- Counter updates do not change the mapping used by the current invocation.

After execution and counter updates, a node with experts on both CPU and CUDA computes:

```text
cpu_max = maximum counter among CPU experts
cuda_min = minimum counter among CUDA experts
```

The node exchanges the corresponding experts only when:

```text
cpu_max > (1 + epsilon) * cuda_min
```

Counter ties use expert ID order. The pilot evaluates placement only after the complete model inference and selects at
most one exchange per node for that boundary. Exchanges are fully asynchronous with respect to the next inference.

Each exchange uses separate incoming and outgoing pinned host buffers, one shared CUDA layout buffer, and an extra
CUDA staging slot so the published mapping remains usable until the complete exchange is ready:

1. Record completion of all CUDA work that references the expert selected for CPU residency.
2. Transpose the outgoing matrix weights to MLAS layout on CUDA and copy them to the outgoing pinned buffer.
3. Stage the selected CPU expert in incoming pinned memory. After the outgoing transfer finishes, upload the incoming
   data and transpose it on CUDA into the extra CUDA slot.
4. Record a transfer-completion event.
5. Commit the outgoing expert to its CPU allocation and publish the new mapping only after both directions have completed.

The CPU-bound transfer completes before the CPU-to-CUDA transfer starts because both directions reuse the CUDA layout
buffer; this is an implementation constraint, not a restriction imposed by NVIDIA copy engines. An exchange never
overwrites the currently published CUDA slot. If the next inference reaches the same `MoE` before the exchange completes,
the kernel uses the pre-exchange mapping and does not wait. A later invocation observes the new placement only after
the pilot has published the completed exchange.

```text
inference t
    -> every offloaded MoE executes with immutable placement P
    -> counters are updated
    -> after Run() completes, the pilot selects and enqueues exchanges

inference t+1, node L
    -> if L's exchange is incomplete, execute with placement P without waiting
    -> if L's exchange is complete, publish and execute with placement P+1
```

CUDA devices expose their copy-engine count, but that value does not provide a portable expert-level concurrency
guarantee. The initial implementation therefore permits at most four in-flight expert exchanges per CUDA device.
Separate pinned buffers, CUDA staging slots, and copy streams allow transfers from different exchanges to overlap.
Within one exchange, the incoming transfer waits for the outgoing transfer before reusing the shared CUDA layout
buffer. Publication waits for both directions. Hardware with one copy engine serializes the transfers without changing
correctness. Additional eligible exchanges are reconsidered at a later inference boundary, not retained in a queue.
For dense MoE, the two expert-sized pinned buffers and two expert-sized CUDA staging/layout buffers exist only while
an exchange is pending and are released after its publication event completes. QMoE currently performs the packed
expert exchange synchronously at the inference boundary with two reusable pinned staging buffers.

CPU routing processes at most 256 input rows per chunk. Each node retains and reuses compact pinned-host and CUDA
activation buffers sized for one chunk, so long prefills do not establish a full-row-count high-water allocation in
every MoE node. Dense MoE overlaps the first chunk with CUDA expert computation; QMoE currently processes its chunks
synchronously.

## Planned end-of-inference redistribution

Redistribution across nodes is not implemented. A future policy could recompute how many CUDA experts belong to each
node after complete model inference, while preserving the global CUDA budget. Current adaptive swaps only exchange
experts within a node and never transfer CUDA slots between nodes.

The allocation objective is lexicographic:

1. Maximize the number of nodes whose complete expert set is CUDA-resident.
2. Among allocations with the same number of complete CUDA nodes, maximize the sum of counters retained on CUDA.
3. Break remaining ties by node index and expert ID.

Within each node, keep the experts with the highest counters. Redistribution may transfer slot ownership between
nodes, whereas a per-node exchange changes the expert stored in a slot without changing that node's slot count.

The future redistribution policy must not drain or wait for pending exchanges. A node with an incomplete exchange
must continue using its published pre-exchange placement. Slot metadata must change only after both transfer directions
complete.

## Operator integration

Routing instrumentation covers `com.microsoft::MoE` and `com.microsoft::QMoE`. CPU offload and adaptive swaps currently
cover only built-in CUDA FP16/BF16 `MoE`, without changing its exported schema or inserting a new graph operator.

The graph node remains assigned to the CUDA execution provider. Its CUDA kernel:

- owns or accesses the runtime expert cache;
- dispatches resident experts to CUDA;
- invokes shared CPU expert-compute helpers for CPU-resident experts;
- submits counter updates and, once adaptive swaps are enabled, placement changes to the cache manager.

Session initialization uses the constant-weight prepacking path to materialize the CPU and CUDA subsets in their
respective compact stores. Constant FC1/FC2 weights and any connected biases, enabled prepacking, and a statically known
hidden dimension are required. With CPU offload disabled, the existing dynamic-weight CUDA path is unchanged.

Prepacking retains the initializer's shared owner rather than copying every expert into a CPU vector. For external
weights, this preserves a file-backed mapping without materializing the complete model on CPU. Each node's selected
CUDA experts are gathered directly into their compact CUDA store through the existing external-data loader's two
64 MiB pinned upload buffers. Only the CPU subset is converted into CPU GEMM layout, and the original initializer is
released when that node's stores are ready. Disabling the CUDA external-data loader retains synchronous per-expert
uploads without restoring a complete CPU staging copy. Inline or user-supplied weights keep their existing owner until
initialization completes; this does not remove memory already occupied by the supplied model.

When `session.moe_cpu_offload_experts` is `0` or absent, CPU and CUDA `MoE`/`QMoE` behavior remains unchanged.

## State ownership and concurrency

The root `SessionState` owns a dedicated `KernelPilotMoeExpertState` shared with its subgraph session states. It contains:

- policy parameters;
- per-node expert counters;
- the global expert budget and per-node allocation targets.

CPU and CUDA kernels obtain their session-owned `KernelPilot` through `OpKernelContext::GetKernelPilot()`.
The generic pilot in `core/framework` carries no kernel-specific logic; `KernelPilotMoeExpertSelection`, defined in
`kernel_pilot_moe_expert_selection.h`, deduplicates selected expert IDs. CUDA-specific copies and events stay in a
provider-side adapter.
After successful kernel execution, the executor records usage through
`KernelPilotMoeExpertState::RecordUsage()`. CPU usage is committed directly; CUDA-offloaded usage is deferred until
`EndRun` consumes the provider snapshot and updates the host mirror. Failed kernel invocations are drained without
committing their counters, while earlier successful invocations still commit if later run work fails. Swaps are
scheduled only when the overall run succeeds. These are internal C++ calls, not a public C API.
Expert identity includes graph scope as well as node and expert IDs, so nodes in different
subgraphs cannot collide. The state persists across `Run()` calls and is isolated from other sessions.

Each CUDA kernel cache owns device-specific resources and execution state:

- CUDA slots, one reusable staging slot, one shared layout buffer, and the current immutable mapping;
- separate reusable incoming and outgoing pinned host staging buffers;
- at most one pending exchange;
- CUDA completion events.

Initialization builds an immutable dictionary from `(OpKernel pointer, local expert ID)` to a global expert index.
Counters are ordinary values; their update path does not acquire a mutex when logging is disabled. Counting and adaptive
placement require non-overlapping `Run()` calls on a session, enforced at run entry. Distinct kernels may update disjoint
expert ranges, but a kernel cannot execute twice simultaneously. Global snapshots are read between runs.
Each invocation captures a stable placement mapping, and a slot cannot be overwritten until all work using its
previous contents has completed.

Errors are explicit. Invalid configuration, invalid initial state, allocation failures, copy failures, and event
failures fail session initialization or execution rather than silently disabling offload or retaining stale placement.

## Implementation scope

The combined implementation includes static hybrid inference and adaptive placement in one change. Static placement
remains available by setting `session.moe_max_in_flight_swaps_per_device=0`; enabling swaps does not change the operator
schema or the session-global CPU expert count.

### Step 1: static FP16/BF16 placement and hybrid inference

Implemented for the built-in CUDA FP16 and BF16 `MoE` paths:

- Parse and validate `session.moe_cpu_offload_experts`; `0` disables offloading.
- Use loaded counters to select CUDA experts globally; distribute an all-zero budget round-robin across eligible nodes.
- Keep only CPU-resident experts in compact GEMM-layout CPU storage and materialize CUDA-resident experts in compact
  CUDA storage. A swap exchanges one expert between these fixed-capacity stores through reusable staging buffers.
- Keep this initial placement immutable: this step has no swaps or end-of-inference redistribution.
- Dispatch CUDA-resident routes through CUTLASS. CPU-resident FP16 routes use MLAS FP16 GEMMs; BF16 weights are
  converted once during initialization and CPU-resident BF16 routes use MLAS FP32 GEMMs.
- Complete the host input copy before CUDA expert kernels can overwrite an aliased input/output buffer, then overlap
  CPU and CUDA expert GEMMs.
- Combine CPU and CUDA expert results on CUDA without changing the exported `MoE` model contract.
- Continue collecting complete routing selections and updating counters.
- Preserve dynamic FP16/BF16 weights and the existing CUDA implementation when offloading is disabled.
- Test CPU-only, CUDA-only, and mixed expert execution, numerical agreement, counter updates, invalid configuration,
  and unchanged disabled-path behavior.

Current limitations are deliberate: constant FC1/FC2 weights are required; FC3, sparse mixer, QMoE, FP32,
minimal builds, CUDA graph capture, and the CUDA plugin EP are not supported by this first slice.
Prepacking must remain enabled, the input hidden dimension must be statically known, and connected FC1/FC2 biases
must also be constant.

### Step 2: adaptive expert swaps

Implemented in the adaptive-swap change:

- Use the session-global counters and optional initial counter state to rank experts.
- Apply the strict `cpu_max > (1 + epsilon) * cuda_min` rule and let the pilot schedule exchanges after inference.
- Move the CUDA expert to pinned CPU staging before moving its replacement to the extra CUDA slot.
- Use dedicated device-to-host and host-to-device streams, events, and separate staging allocations for each active
  exchange.
- Permit at most four in-flight exchanges per CUDA device; additional eligible nodes are reconsidered at the next
  inference boundary.
- Pipeline the host-to-device transfer of one exchange with the device-to-host transfer of another when the hardware
  supports bidirectional copies.
- Never wait for an incomplete exchange when a `MoE` starts. The old placement remains published until transfer
  completion; publication copies the staged expert into the persistent slot on that invocation's compute stream.
- Preserve each node's CUDA slot count and therefore the session-global CUDA expert budget.
- Test the strict epsilon boundary, deterministic tie-breaking, multi-exchange concurrency, delayed
  publication, and numerical parity before and after a swap.

Still planned:

- Redistribute CUDA slot ownership across nodes without draining pending exchanges, while maximizing the number of
  completely CUDA-resident nodes.
- Extend end-to-end throughput and memory measurements to other models and hardware.

This change includes the warmed CUDA-only and hybrid parameter measurements below, a reproducible timing script and
aggregate results. Broader model-quality, CPU-only and multi-architecture evaluation remains future work. Oversized
raw traces stay outside the repository.

## Performance evaluation

Measure:

- token throughput;
- peak CUDA memory;
- peak CPU memory.

### Reproducible warmed-decode protocol

[`benchmark_moe_cpu_offload.py`](../../tools/python/benchmark_moe_cpu_offload.py) uses the existing GenAI environment
and a source-built ORT runtime. On Linux, `ORT_PYTHON_ROOT` must expose an `onnxruntime` package whose `capi` directory
contains the matching shared runtime and CUDA provider libraries, or symlinks to them. GenAI locates these libraries
through the package found on `PYTHONPATH`; selecting the right Python executable alone does not select the right ORT
binary. Verify the resolved `ORT_LIB_PATH` after importing GenAI and record the runtime/provider hashes.

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$ORT_PYTHON_ROOT:tools/python" \
python tools/python/benchmark_moe_cpu_offload.py \
  --name cpu1000-alpha0p9-epsilon0p1 \
  --model "$MODEL_PATH" --prompts docs/moe-cpu-offload/benchmark-prompts.json \
  --prompt-limit 5 --max-new-tokens 256 \
  --cpu-offload-experts 1000 --counter-alpha 0.9 --swap-epsilon 0.1 \
  --swap --no-counter-log --no-cuda-graph \
  --output result.json --log result.log
```

The bounded 256-row activation-chunk implementation was measured on 2026-10-14 with that exact command and one
fresh process. The prompt 3-5 score was **20.870 ms/token** (individual prompt means: 21.075, 18.632, and
22.901 ms/token), compared with **21.557 ms/token** for the earlier two-run reference. Model loading took 55.190 s;
GPU memory was 60.53 GiB after loading and peaked at 61.86 GiB, while process RSS peaked at 7.64 GiB. The measured
runtime and CUDA provider SHA-256 values were `29ee22201693e482fa798551dd61972827eb57bcf48b43102c365ff4e37d7a3a`
and `14ce18c4b1d35d8c14c9198e7cfbf08372f7ce114200cc2b7ab9e25a51362831`, respectively.

Each fresh process runs five prompts, with counters and placement retained between prompts. Generation is greedy and
has a fixed length of 256 tokens. `append_tokens()` is prefill and is timed separately. The first
`generate_next_token()` uses those precomputed logits, so it is excluded from decode timing. Score only prompts 3-5:
each contributes 255 actual decode inferences, or 765 calls per repetition. The reported repetition score is the mean
of those three prompts' `decode.mean_ms` values, not the total generation duration or the best prompt.

Report the actual number of repetitions, mean, sample standard deviation, and range; standard deviation is unavailable
for single-run configurations. The initial three-repetition plan was reduced to one above 1,024 CPU experts, then
stopped early at the author's request to publish available results. The tables below include every completed in-grid
measurement, with one or two repetitions per configuration; no three-repetition precision is claimed.
Run configurations serially on one otherwise idle GPU, interleaving configurations between repetitions to reduce
temporal drift. Do not profile, build, or run other GPU tests during measurements. Memory is sampled every 50 ms;
reported peaks include initialization and all five prompts. GPU memory is device-wide NVML usage, while CPU memory
is process RSS.

The parameter sweeps use alpha `{0.5, 0.75, 0.9, 0.95, 0.99}`, epsilon `{0, 0.01, 0.05, 0.1, 0.2, 0.5}`, and CPU
expert counts `{1, 100, 256, 512, 768, 1024, 1280, 1536, 1792, 2048}`, with `0` retained separately as an all-GPU
reference. The multiples of 256 represent whole-layer-equivalent expert counts, not necessarily whole CPU-resident
layers: the count is session-global and initial placement distributes slots across nodes. Hold the other parameters
at alpha `0.9`, epsilon `0.1`, and 1,000 CPU experts. These are benchmark reference values, not changed session defaults.
The alpha and epsilon tables share the same reference measurements. Including the separate all-GPU reference,
there are 21 unique configurations and 33 completed in-grid runs. Higher-count second repetitions that had already
finished before the repetition limit was requested are included and explicitly counted; no further runs were started.
Probability weighting remains at its default of zero and the swap concurrency limit is four. At zero CPU experts there
is no offload, counter instrumentation, or applicable exchange; that row is not a measurement of active swaps.

### Available parameter measurements (2026-10-13)

Measured source: `63148bd3b96a7c38a15b7b8f265df44b7f0e3a54`, incorporating target main `9036f076a2`.
Hardware: one NVIDIA H200 (SM90, 143,771 MiB, driver 580.105.08), Intel Xeon Platinum 8480C, 96 physical cores,
two sockets/NUMA nodes, no SMT. Software: Linux, Python 3.12, GenAI 0.15.2, ORT 1.32.0, ONNX 1.23.2, CUDA 13.3.73.
Release build with native CUDA architecture, Flash/MEA enabled and quick build disabled. The model is the FP16
Qwen3.5-35B-A3B export; the unchanged five-prompt input list comes from the earlier routing experiments.

The tables report available fresh-process runs, not confidence intervals or a complete three-repeat sweep.
The [CSV](parameter-benchmarks.csv) contains individual repetition scores, aggregate timings, post-load memory and
sampled peaks. Empty repetition/standard-deviation fields mean unmeasured, not zero. No profiler was used; the GPU
was empty before each process, and runtime/provider hashes were unchanged across measurement and publication.

#### Alpha sweep: epsilon 0.1, 1,000 CPU experts

| Alpha | Repetitions | Decode mean +/- sample SD (ms/token) | Run range (ms/token) | GPU after load (GiB) | Peak GPU / CPU RSS (GiB) |
|---|---|---|---|---|---|
| 0.5 | 1 | 19.270 +/- n/a | 19.270-19.270 | 61.03 | 62.91 / 61.27 |
| 0.75 | 1 | 22.099 +/- n/a | 22.099-22.099 | 60.78 | 62.67 / 61.26 |
| 0.9 | 2 | 21.557 +/- 0.706 | 21.058-22.056 | 60.78 | 62.67 / 61.26 |
| 0.95 | 2 | 21.318 +/- 0.862 | 20.709-21.927 | 60.66 | 62.55 / 61.26 |
| 0.99 | 1 | 20.607 +/- n/a | 20.607-20.607 | 61.03 | 62.91 / 61.27 |

#### Epsilon sweep: alpha 0.9, 1,000 CPU experts

| Epsilon | Repetitions | Decode mean +/- sample SD (ms/token) | Run range (ms/token) | GPU after load (GiB) | Peak GPU / CPU RSS (GiB) |
|---|---|---|---|---|---|
| 0 | 2 | 21.552 +/- 1.451 | 20.526-22.578 | 60.54 | 62.43 / 61.26 |
| 0.01 | 1 | 22.544 +/- n/a | 22.544-22.544 | 61.03 | 62.91 / 61.27 |
| 0.05 | 1 | 22.847 +/- n/a | 22.847-22.847 | 60.53 | 62.42 / 61.26 |
| 0.1 | 2 | 21.557 +/- 0.706 | 21.058-22.056 | 60.78 | 62.67 / 61.26 |
| 0.2 | 1 | 21.468 +/- n/a | 21.468-21.468 | 60.53 | 62.42 / 61.26 |
| 0.5 | 2 | 20.233 +/- 0.523 | 19.863-20.602 | 60.65 | 62.54 / 61.26 |

#### CPU expert-count sweep: alpha 0.9, epsilon 0.1

| CPU experts | Repetitions | Decode mean +/- sample SD (ms/token) | Run range (ms/token) | GPU after load (GiB) | Peak GPU / CPU RSS (GiB) |
|---|---|---|---|---|---|
| 0 | 2 | 9.059 +/- 1.127 | 8.263-9.856 | 66.31 | 68.56 / 2.98 |
| 1 | 2 | 9.370 +/- 0.141 | 9.270-9.469 | 66.67 | 68.94 / 61.27 |
| 100 | 2 | 11.558 +/- 0.726 | 11.044-12.071 | 66.05 | 67.94 / 61.27 |
| 256 | 2 | 13.781 +/- 1.045 | 13.042-14.520 | 65.13 | 67.03 / 61.26 |
| 512 | 1 | 14.770 +/- n/a | 14.770-14.770 | 63.38 | 65.28 / 61.26 |
| 768 | 1 | 19.895 +/- n/a | 19.895-19.895 | 62.39 | 64.28 / 61.27 |
| 1024 | 2 | 20.870 +/- 0.727 | 20.355-21.384 | 60.51 | 62.40 / 61.26 |
| 1280 | 2 | 26.078 +/- 1.672 | 24.896-27.261 | 59.01 | 60.90 / 61.26 |
| 1536 | 1 | 28.509 +/- n/a | 28.509-28.509 | 57.64 | 59.53 / 61.26 |
| 1792 | 2 | 44.844 +/- 2.725 | 42.917-46.770 | 56.02 | 57.91 / 61.26 |
| 2048 | 2 | 84.472 +/- 5.616 | 80.502-88.443 | 54.51 | 56.26 / 61.26 |

At 1,000 CPU experts, the reference averages 21.557 ms/token across two runs. Alpha 0.5 gives the lowest observed
in-grid result, 19.270 ms/token, but has only one run; epsilon 0.5 averages 20.233 ms/token across two. These results
do not demonstrate a stable 10 ms/token gain and are not a controlled comparison against a rebuilt pre-optimization
baseline. The session defaults are unchanged.

Increasing CPU residency saves GPU memory but raises latency sharply at the largest counts. These tables were collected
before the bounded-initialization update, when prepacking temporarily copied all expert weights to CPU and produced
roughly 61 GiB CPU RSS peaks. They do not describe the updated initialization path. At the 1,000-expert reference,
post-load CPU RSS was about 6.51 GiB. Peak RSS is not the live compact expert-weight footprint.
These measurements are not a model-quality evaluation, nor a guarantee for other GPUs or NUMA placement.

#### Bounded-initialization memory check

A scaled external-weight reproducer contains 40 MoE nodes with 256 experts each, width 256, and 2.5 GiB of FP16
weights. Each measurement initializes the same model in a fresh native C API process, with CPU EP fallback disabled
and adaptive swaps disabled. These are single-run initialization measurements, not inference benchmarks.

| Initialization path | CPU experts | Peak CPU RSS (GiB) | CPU RSS after load (GiB) | Initialization (seconds) |
|---|---|---|---|---|
| Before bounded initialization | 1 | 2.88504 | 0.358517 | 3.28443 |
| Bounded initialization | 1 | 0.516357 | 0.361359 | 3.22083 |
| Bounded initialization | 1000 | 0.758240 | 0.605530 | 3.64408 |

The matched one-expert measurement reduces peak CPU RSS by about 82%, without changing the live CPU expert footprint.
Mapped source pages, compact CPU weights, upload buffers, and runtime allocations still contribute to RSS; the two
64 MiB buffers are not an absolute bound on process memory. The original 35B model's peak has not been remeasured
after this change because insufficient free GPU memory was available.

#### GPU-packed INT4 QMoE CPU compute

Qwen3.5-35B-A3B INT4, H200 (physical GPU 0), Xeon Platinum 8480C, 1,000 CPU experts,
alpha 0.9, epsilon 0.1, adaptive swaps enabled, CUDA Graph and counter logs disabled:

| CPU expert implementation | Warm decode (ms/token) | Model load (s) | Peak CPU RSS (MiB) |
|---|---:|---:|---:|
| Dense FP16, three prompts x 128 tokens | 53.425 | 41.907 | 27,157.7 |
| GPU-packed MLAS Q4 FP32, three prompts x 128 tokens | 14.208 | 19.877 | 18,703.5 |
| Dense FP16, five prompts x 256 tokens | 26.294 | 41.944 | n/a |
| GPU-packed MLAS Q4 FP32, five prompts x 256 tokens | 12.434 | 16.897 | n/a |

These are single-process runs, not repeated averages. The three-prompt scores use prompts 2-3; the five-prompt
scores use prompts 3-5. Model load, prefill, the generation call using prefill logits, and the first actual decode
call are excluded. The three-prompt result improves by 3.76x; the five-prompt result improves by 2.11x.
Prefill is still executed and measured separately: the Q4 three-prompt run took 1,302.2, 44.3, and 25.4 ms.
Greedy generation is not replay of an identical routing trace, and this is not a model-quality evaluation.

Both initial placement and outgoing swaps pack weights on GPU directly from the SM80 CUTLASS representation.
The CPU consumes packed INT4 through MLAS QNBit, without CPU weight prepacking or dense FP16 dequantization.
QNBit can dequantize temporary tiles internally for multi-row GEMMs; it does not retain dense expert matrices.
A separate Nsight capture of the three-prompt workload recorded 4,402 GPU packing launches (31.55 ms total device
execution) and CPU samples in `SQ4BitGemmM1Kernel_CompFp32`. No dense FP16 GEMM leaf samples were observed.
The capture includes initialization and inference; its instrumented timings are not used in the table.
The five-prompt protocol is reproducible with `tools/python/benchmark_moe_cpu_offload.py`,
`docs/moe-cpu-offload/benchmark-prompts.json`, `--max-new-tokens 256 --cpu-offload-experts 1000
--counter-alpha 0.9 --swap-epsilon 0.1 --swap --no-counter-log --no-cuda-graph`.
