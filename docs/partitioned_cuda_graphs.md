# Experimental CPU offloading with partitioned CUDA graphs

The existing CUDA graph path replays the session's device work without executing
the CPU executor. Consequently it cannot support offloaded CPU computation inside
the same session. Shape-only CPU nodes are a limited exception, not CPU offloading.

The opt-in partitioned execution prototype retains the original ONNX model and
uses its normal CPU/CUDA placement. It walks the partitioned graph sequentially,
runs CPU nodes and device copies eagerly, and captures contiguous CUDA compute
partitions separately. This supports CPU prefixes, CPU suffixes, and CPU computation
between multiple CUDA partitions. It does not capture CPU computation or move
weights between devices during a run.

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
Only operators with CPU kernels can be offloaded. Existing capacity estimates do
not include the additional retained capture buffers, so reserve memory headroom
rather than treating the partitioning budget as a peak-VRAM guarantee.

Use IOBinding to supply inputs on their consuming devices and request outputs on
their producing devices. Keep input and preallocated output OrtValues alive, with
unchanged addresses, shapes, and types for each `gpu_graph_id`. Updating their
contents is supported. CPU embedding inputs can therefore contain different token
IDs on every invocation. Device copies between model nodes are handled internally.

The first invocation for each graph ID performs two warm-up passes and one capture
pass. A retained execution frame keeps intermediate tensors alive. CUDA scratch
allocations are recorded during the second warm-up, retained, and reused at the
same addresses during capture. Later invocations execute CPU partitions and copies
again, but replay the CUDA partitions instead of invoking their kernels.

`gpu_graph_id=-1` uses the ordinary eager executor, allowing uncaptured prefill.
Different nonnegative IDs can represent different fixed-shape decode buckets.
Every captured bucket retains its own intermediates and scratch until session
destruction. Captured GPU allocations are isolated from eager runs and other
buckets.

## Prototype constraints

- Built-in CUDA EP only, with its default BFC arena; plugin EPs and shared or
  external allocators are not supported.
- Sequential inference only; captured invocations must use the same host thread.
  Control flow, asynchronous host kernels, partial execution, and per-run stream
  overrides are not supported.
- CPU inputs consumed *directly* by CUDA kernels as host-side control data must
  retain their captured values. Changes return an error requesting a new graph ID.
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
  failure, recreate the session. Input binding validation errors before execution
  do not invalidate already captured buckets.

This is not a claim that a particular 27B model fits in 12 or 24 GB. That requires
measurement with its actual quantization, context length, KV cache, placement, and
capture buckets. CPU offloading may also impose substantial decode latency.

## Validation targets

Framework coverage checks scratch address retention, allocation-sequence errors,
and mixed CPU/CUDA execution with two CUDA partitions separated by CPU work.
Changing input values between replays must change the output, and the CUDA EP must
report that both partition graphs were captured. Additional buckets and eager
execution between replays must preserve the original capture.

Real-model evaluation must compare eager and captured output parity, confirm
partition replay in logs, and measure peak VRAM and prefill/decode latency. A
configured `enable_cuda_graph` option alone does not prove that replay occurred.
