# Experimental CPU offloading with partitioned CUDA graphs

The existing CUDA graph path replays the session's device work without executing
the CPU executor. Consequently it cannot support offloaded CPU computation inside
the same session. Shape-only CPU nodes are a limited exception, not CPU offloading.

The opt-in partitioned execution prototype retains the original ONNX model and
uses its normal CPU/CUDA placement. At initialization, it selects the existing
whole-session capture/replay path if placement satisfies the CUDA EP's normal
capture policy and every CPU node is an ONNX Shape or Size node. This includes
all-CUDA graphs, eligible CPU shape nodes without device-copy nodes, and empty
graphs. This path preserves the user's memory-pattern
setting and normal capture/replay behavior; it does not allocate partition-specific
retained frames or scratch. Other CPU computation selects partitioned execution
even if it forms an independent branch with no device copies.

Otherwise, it walks the partitioned graph sequentially, runs CPU nodes, device
copies, and CUDA nodes producing host outputs (such as Shape) eagerly, and captures
contiguous CUDA compute partitions separately. This supports CPU prefixes, CPU
suffixes, and CPU computation between multiple CUDA partitions. It does not capture
CPU computation or move weights between devices during a run.

Selection is based on the finalized graph placement, not a runtime capture attempt.
Unsupported control flow is still rejected; capture errors do not silently trigger
a switch between execution modes. CPU shape nodes allowed by whole-session capture
retain its existing requirement that their results remain valid across replays.

## Configuration

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_session_config_entry("session.enable_partitioned_cuda_graph", "1")
options.add_session_config_entry("session.partitioned_cuda_graph_max_ids", "16")
options.add_session_config_entry(
    "session.name_based_layer_assignment",
    "cpu(embed_tokens);gpu(layers.,lm_head)",
)

session = ort.InferenceSession(
    "model.onnx",
    options,
    providers=[
        ("CUDAExecutionProvider", {"enable_cuda_graph": "1"}),
        "CPUExecutionProvider",
    ],
)
```

Match placement rules to the actual node names. Annotation-based placement and
capacity-aware partitioning remain available; see
[partitioning with annotations and memory constraints](annotated_partitioning/PartitioningWithAnnotationsAndMemoryConstraints.md).
The prototype changes execution, not placement policy or memory estimation.
Both the built-in CUDA EP and the CUDA plugin EP are supported. For a plugin build,
register/select the CUDA plugin as usual and set the same session option and
`enable_cuda_graph` provider option; no new plugin C API is required.
Only operators with CPU kernels can be offloaded. Existing capacity estimates do
not include the additional retained capture buffers, so reserve memory headroom
rather than treating the partitioning budget as a peak-VRAM guarantee.

Use IOBinding to supply inputs on their consuming devices and request outputs on
their producing devices. Keep input and preallocated output OrtValues alive, with
unchanged addresses, shapes, and types for each `gpu_graph_id`. Updating their
contents is supported. CPU embedding inputs can therefore contain different token
IDs on every invocation. Device copies between model nodes are handled internally.

When partitioned execution is selected, the first invocation for each graph ID
performs two preparation passes to allocate outputs and retain scratch. The built-in
EP then captures on the next pass. The plugin runs these preparation passes with
capture disabled, then honors its `min_num_runs_before_cuda_graph_capture` setting
with up to eight further passes (including capture). Values from zero through seven
are supported; a larger count returns an explicit capture-attempt-limit error.
A retained execution frame keeps intermediate tensors alive. Device scratch
allocations are intercepted in ORT's arena or plugin arena adapter and reused at the
same addresses during capture. Pinned host staging buffers are excluded from this
interception: their release is deferred until stream cleanup during preparation,
and the CUDA EP retains buffers referenced by captured copies for replay.
Stream wrappers are borrowed from the session's per-thread pool for each invocation,
so capture buckets, eager runs, and IOBinding input copies reuse the same cleanup
owner for the CUDA plugin's graph stream. Retained frames detach from those wrappers
before they return to the pool; no bucket reserves a separate stream collection.
Later invocations execute CPU partitions, CUDA host-output nodes, and copies again,
but replay the CUDA compute partitions instead of invoking their kernels.

`gpu_graph_id=-1` uses the ordinary eager executor, allowing uncaptured prefill.
Different nonnegative IDs can represent different fixed-shape decode buckets.
IDs below `-1` are rejected before allocating partitioned capture state or executing
any nodes, without invalidating existing buckets.
Every captured bucket retains its own intermediates and scratch until session
destruction. Captured GPU allocations are isolated from eager runs and other
buckets.

`session.partitioned_cuda_graph_max_ids` limits the number of retained nonnegative
graph IDs per session. It defaults to **16** and accepts a decimal integer from 1
through 2147483647; zero does not mean unlimited. Once the limit is reached, an
unseen ID is rejected before constructing its retained execution state or running
any partitions. This rejection does not invalidate the session: existing IDs and
eager runs remain usable. Reusing an ID does not consume another slot.
The limit counts user-visible buckets, not the internal CUDA graphs within each
bucket, and is not a CPU/GPU byte budget. Choose a lower limit for large captures.
There is no eviction or per-ID retirement; recreate the session to release all
buckets or change the limit. Whole-session capture and ordinary eager execution
ignore this setting.

## Returned output ownership

Automatically allocated outputs are **mutable per-ID views, not independent
snapshots**. The retained frame keeps owning OrtValue references for the session's
lifetime, and returned OrtValues share those tensors. A later partitioned run with
the same graph ID reuses their storage, including CPU outputs. Keeping an earlier
OrtValue alive therefore does not preserve its contents: it observes later writes,
even when the next run uses a fresh empty fetch vector or newly created IOBinding.
A failed partitioned run may also leave these views partially updated.

Copy results that must be preserved into independent storage before another run
reuses that ID. CPU results can be copied on the host; device results require an
explicit host/device copy. Rebinding a different output buffer to the same ID is
not a snapshot mechanism: preallocated output addresses must still match the
captured bindings.

## Prototype constraints

The option requires a non-minimal build, a CUDA EP with capture enabled, and an ONNX
model without control flow. Minimal builds exclude the partition-capture machinery
and reject enabling this option. The additional constraints below apply when partitioned
execution is selected; eligible whole-session graphs follow the existing CUDA
capture requirements instead.

- A CUDA arena accessed through the session's allocator is required; shared
  environment or external allocators are not supported. Plugin kernel scratch must
  use the allocator returned through ORT's kernel APIs so retention covers the
  allocations across the plugin boundary.
  The plugin must implement run-start/run-end and synchronization callbacks in
  addition to capture/replay.
- Sequential inference only; captured invocations must use the same host thread.
  Control flow, asynchronous host kernels, partial execution, and per-run stream
  overrides are not supported.
- All node outputs must be tensors. Sequence, map, optional, and sparse outputs are
  rejected at initialization, including CPU-only intermediates. Their kernels may
  require fresh output objects, which the retained execution frame cannot provide.
- CPU inputs consumed *directly* by CUDA kernels as host-side control data must
  retain their captured values. Select a new graph ID before changing them. A
  mismatch detected during partition replay invalidates the session because earlier
  partitions may already have updated outputs or in-place state.
  This differs from CPU tensor data transferred through a `MemcpyFromHost` node,
  which is refreshed on every invocation.
- CUDA kernels producing host outputs run eagerly and synchronize before subsequent
  CPU work. Their outputs, including pinned host tensors, are checked as host control
  inputs when consumed directly by a captured partition. Shape/data-dependent changes
  in tensor allocations or scratch allocation sequences are rejected rather than
  silently replaying stale parameters.
- Memory patterns are disabled. Intermediates and scratch remain resident, and
  scratch allocation reuse is deliberately conservative. Capturing many buckets
  may substantially increase memory consumption even within the configured ID limit.
- Partition replay and device copies synchronize at boundaries. This establishes
  correctness but does not overlap transfers and computation.
- Capture warm-up executes CPU computation repeatedly, so stateful CPU operators
  are not an appropriate workload.
- Individual graph retirement is not implemented. If a partitioned invocation
  fails after execution/capture begins, including termination between partitions,
  recreate the session.
  All subsequent runs are rejected, including eager `gpu_graph_id=-1` calls,
  before provider run callbacks or node execution. Input binding validation errors
  and termination detected before partitioned execution do not invalidate the
  session; existing buckets and eager execution remain usable.
  An error originating in ordinary eager `gpu_graph_id=-1` execution does not set
  this partitioned failed-state flag; ordinary eager error handling still applies.
  This is distinct from rejecting an eager call after a partitioned failure.

This is not a claim that a particular 27B model fits in 12 or 24 GB. That requires
measurement with its actual quantization, context length, KV cache, placement, and
capture buckets. CPU offloading may also impose substantial decode latency.

## Observability

Preparation/capture passes and eager CPU, copy, and CUDA host-output nodes use the
standard sequential executor's kernel wrapper without recycling frame values.
Session-level and run-level profiling record per-node events and allocator
statistics for these invocations. Node allocation statistics, memory-profiler
hooks, NVTX ranges, debug input/output hooks, and node-specific error attribution
are shared with ordinary execution.

CUDA partition replay launches an existing graph without re-invoking its kernels.
It therefore produces no synthetic per-node kernel events. Use a CUDA trace to
measure graph launches rather than interpreting preparation/capture host timings
as replay timings. Eager allocation statistics do not describe the extra retained
memory of partitioned capture.

MoE statistics remain incompatible with graph capture, so this path must not be
used to bypass that exclusion. Adaptive CPU/GPU MoE routing would violate fixed
placement and capture/allocation-sequence assumptions. Supporting it would require
a separate design, such as an eager orchestrator around fixed CUDA sub-operations.

## Validation targets

Framework coverage checks scratch address retention, allocation-sequence errors,
and mixed CPU/CUDA execution with two CUDA partitions separated by CPU work.
Changing input values between replays must change the output, and the CUDA EP must
report that both partition graphs were captured. Additional buckets and eager
execution between replays must preserve the original capture.
Unequal-length CUDA Concat coverage checks pinned host staging buffers across
two CUDA partitions, changing CPU inputs, different-shape buckets, and eager runs.
It also verifies reuse of the same pooled stream collection and wrappers after each
run, including repeated IOBinding input rebinding between buckets and eager execution.
CUDA Shape coverage checks that host outputs are recomputed between captured
partitions rather than skipped during replay.
Routing coverage also checks all-CUDA graphs, CPU shape nodes without device
copies, and empty graphs. Eligible graphs must capture under the user-supplied graph
ID instead of the partition executor's internal IDs, and preserve enabled or disabled
memory-pattern settings. Mixed-device graphs must still select partitioned execution.
The same routing and mixed-device tests run with the CUDA plugin, with additional
coverage for configurable warm-up counts and scratch retention across the allocator
C ABI boundary. Failure regressions check that scratch retained by a kernel during
a rejected replay is freed only by that kernel, and that partial replay failures
invalidate every captured bucket while pre-execution validation remains recoverable.
Post-start partitioned failures also reject eager execution, including IOBinding
runs, without modifying output buffers. Coverage includes partial termination,
changed host control inputs, and capture-attempt exhaustion. An eager-origin kernel
failure followed by successful captured replay checks the distinct eager failure
contract.
Fresh automatically allocated fetches and IOBinding outputs reuse same-ID CPU
result storage: an earlier retained OrtValue observes the later writes, while an
independent copy preserves its earlier contents.
Graph-ID limit coverage checks the default and configured bounds, invalid settings,
repeated rejection without new device allocations or changed outputs, and continued
replay/eager execution at capacity. Whole-session routing remains exempt from the limit.
Independent CPU-branch coverage verifies changing CPU data without device-copy
nodes, session/run profiling during preparation and replay, and node-specific
failure attribution. Invalid negative graph IDs are rejected before and after
filling the bucket limit without allocating device memory or changing outputs.
MoE counting/statistics remain rejected with capture enabled, with or without the
partitioned option.

Real-model evaluation must compare eager and captured output parity, confirm
partition replay using a CUDA trace or replay logs, and measure peak VRAM and
prefill/decode latency. A configured `enable_cuda_graph` option alone does not prove
that replay occurred.
