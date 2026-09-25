# Experimental CPU offloading with partitioned CUDA graphs

The existing CUDA graph path replays the session's device work without executing
the CPU executor. Consequently it cannot support offloaded CPU computation inside
the same session. Shape-only CPU nodes are a limited exception, not CPU offloading.

The opt-in partitioned execution prototype retains the original ONNX model and
uses its normal CPU/CUDA placement. At initialization, it selects the existing
whole-session capture/replay path if placement satisfies the CUDA EP's normal
capture policy. This includes all-CUDA graphs, eligible CPU shape nodes without
device-copy nodes, and empty graphs. This path preserves the user's memory-pattern
setting and normal capture/replay behavior; it does not allocate partition-specific
retained frames or scratch.

Otherwise, it walks the partitioned graph sequentially, runs CPU nodes and device
copies eagerly, and captures contiguous CUDA compute
partitions separately. This supports CPU prefixes, CPU suffixes, and CPU computation
between multiple CUDA partitions. It does not capture CPU computation or move
weights between devices during a run.

Selection is based on the finalized graph placement, not a runtime capture attempt.
Unsupported control flow is still rejected; capture errors do not silently trigger
a switch between execution modes. CPU shape nodes allowed by whole-session capture
retain its existing requirement that their results remain valid across replays.

## Configuration

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_session_config_entry("session.enable_partitioned_cuda_graph", "1")
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
A retained execution frame keeps intermediate tensors alive. Scratch allocations
are intercepted in ORT's arena or plugin arena adapter and reused at the same
addresses during capture. Later invocations execute CPU partitions and copies
again, but replay the CUDA partitions instead of invoking their kernels.

`gpu_graph_id=-1` uses the ordinary eager executor, allowing uncaptured prefill.
Different nonnegative IDs can represent different fixed-shape decode buckets.
Every captured bucket retains its own intermediates and scratch until session
destruction. Captured GPU allocations are isolated from eager runs and other
buckets.

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
- CUDA kernels producing host outputs are rejected. Shape/data-dependent changes
  in tensor allocations or scratch allocation sequences are rejected rather than
  silently replaying stale parameters.
- Memory patterns are disabled. Intermediates and scratch remain resident, and
  scratch allocation reuse is deliberately conservative. Capturing many buckets
  may substantially increase memory consumption.
- Partition replay and device copies synchronize at boundaries. This establishes
  correctness but does not overlap transfers and computation.
- Capture warm-up executes CPU computation repeatedly, so stateful CPU operators
  are not an appropriate workload.
- Individual graph retirement is not implemented. After an execution/capture
  failure, including termination between partitions, recreate the session. Input
  binding validation errors and termination detected before execution do not
  invalidate already captured buckets.

This is not a claim that a particular 27B model fits in 12 or 24 GB. That requires
measurement with its actual quantization, context length, KV cache, placement, and
capture buckets. CPU offloading may also impose substantial decode latency.

## Validation targets

Framework coverage checks scratch address retention, allocation-sequence errors,
and mixed CPU/CUDA execution with two CUDA partitions separated by CPU work.
Changing input values between replays must change the output, and the CUDA EP must
report that both partition graphs were captured. Additional buckets and eager
execution between replays must preserve the original capture.
Routing coverage also checks all-CUDA graphs, CPU shape nodes without device
copies, and empty graphs. Eligible graphs must capture under the user-supplied graph
ID instead of the partition executor's internal IDs, and preserve enabled or disabled
memory-pattern settings. Mixed-device graphs must still select partitioned execution.
The same routing and mixed-device tests run with the CUDA plugin, with additional
coverage for configurable warm-up counts and scratch retention across the allocator
C ABI boundary. Failure regressions check that scratch retained by a kernel during
a rejected replay is freed only by that kernel, and that partial replay failures
invalidate every captured bucket while pre-execution validation remains recoverable.

Real-model evaluation must compare eager and captured output parity, confirm
partition replay in logs, and measure peak VRAM and prefill/decode latency. A
configured `enable_cuda_graph` option alone does not prove that replay occurred.
