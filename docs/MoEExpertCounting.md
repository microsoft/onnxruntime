# Session-global MoE expert counters

Expert usage counters are enabled only when CPU offload is enabled:

```text
session.moe_cpu_offload_experts=<positive expert count>
```

Without offload, counters and routing snapshots are not allocated, state files are not loaded, and popularity files
are not written. Requests for counting, statistics, or popularity output emit a warning and do not enable counters.
This applies to both CPU and CUDA sessions. Nodes assigned to the CPU execution provider never participate in
offload counters; CPU experts within a CUDA-offloaded node are still counted by that node.
Only offload-eligible CUDA FP16/BF16 nodes are registered, so FP32 CUDA MoE nodes in a mixed model are excluded
from both the counters and the logged node/expert totals.
Prepacked INT4/INT8 CUDA QMoE nodes likewise retain ordinary GPU execution without allocating offload or
counter-collection resources, even when other nodes in the same session use CPU offload.
`session.enable_moe_expert_statistics=1` emits one structured
`moe_expert_counters` log after each successful MoE kernel invocation, subject to the per-Run logging limit below.
Configure the update with:

```text
session.moe_expert_counter_alpha=<finite value in [0, 1]> # default: 0.75
session.moe_expert_counter_probability_weight=<finite value in [0, 1]> # default: 0
```

The selected-expert increment is `1 - alpha`. With CPU offload, the optional probability weight `lambda` changes the
selected contribution from `1` to `(1 - lambda) + lambda * routing_probability`.

It supports built-in CUDA `MoE` and `QMoE` kernels with CPU offload. Minimal builds, the CUDA plugin EP, and CUDA graph
capture are not supported when counting, counter-update logging, or CPU offload is enabled.
Minimal builds omit the counter state, its initialization/run bookkeeping, and the CUDA routing-snapshot collector
sources. They reject counter configuration except for explicitly disabling counting.

Each node invocation updates every expert counter belonging to that node:

```text
c(t+1) = alpha * c(t) + (1 - alpha) * contribution
```

By default, `contribution` is `1` for a selected expert and `0` otherwise. With probability weighting, repeated
selections use the maximum routing probability observed for that expert during the invocation. Unused experts still
receive the decay term.
Counts persist across `Run()` calls. Only successful kernel invocations commit their collected usage.
A failed run can retain updates from earlier successful kernels, including CUDA nodes whose
expert computations were still in flight when their counters were updated. Updates are not transactional across an
entire model invocation.
Provider exceptions during deferred usage completion become failure statuses, and run tracking is released even on
failure. Cleanup errors on an already-failed run are logged without replacing its original error or escaping the
cleanup guard.

The root `SessionState` owns a `KernelPilotMoeExpertState` shared with its subgraph states. Sessions never share counters.
While counter-update logging is active, the state also stores the current request ID and logger. Because expert
selections and counters are session-owned mutable state, a concurrent Run is rejected whenever counting or
counter-update logging is enabled, or when CPU offload uses the counters.
Registration uses graph scope and resolved node index, with one counter per expert. The router's expert dimension must
be statically known when the session is initialized. The global expert count is the sum of those per-node dimensions;
for `N` equally sized MoE nodes with `E` experts, it is `N * E`.

After kernel creation, initialization builds an immutable dictionary
`(OpKernel pointer, local expert ID) -> global expert index`, plus the contiguous expert range for each kernel.
Counters are ordinary `double` values in a session-wide array. `RecordUsage()` uses the kernel pointer and this
dictionary directly, without atomic counters, graph-name lookup, or per-invocation allocation. When logging is enabled,
the same method emits one `moe_expert_counters` JSON record containing the request ID, node identity, deduplicated
selected experts, and the updated counters. The node identity includes its graph scope so equal node indices in
different subgraphs remain distinguishable.
Records include `expert_offset`, the beginning of the node's range in the runtime's global expert array.
Counter replay uses this offset for deterministic tie-breaking, independently of layer names or execution order.
Legacy main-graph traces without offsets use resolved node-index registration order; replay rejects legacy subgraph
traces without offsets rather than guessing their global order.
Logging is limited to 1024 counter records per `Run()`, shared across all nodes and subgraphs.
On the first omitted record, a single WARNING `moe_expert_counters_truncated` JSON marker reports
`request_id` and `max_records`. Later records are omitted without formatting their JSON.
Counter updates continue unchanged after the limit, and the logging budget resets on the next Run.
The Python analyzer rejects any trace containing this marker, even with an otherwise valid completion footer.
When INFO output is disabled, counter records are neither formatted nor charged against the logging budget.
Only logging-budget reservation uses an atomic; counting-only updates remain unsynchronized across disjoint kernel ranges.

`tools/python/qmoe_expert_activation_correlations.py` analyzes complete counter traces as binary expert activations.
For every MoE node, it computes the full expert-by-expert Pearson correlation matrix against the same node at the
previous decode token. For every layer after the first, it also computes the matrix against the preceding layer at
the same token. Prefill records are excluded, decode invocations are aligned independently within each prompt, and
previous-token pairs never cross prompt boundaries. The output contains compressed NumPy matrices for all expert
pairs and their activation counts, a PDF of heatmaps, and a CSV listing the 20 highest and 20 lowest sufficiently
supported correlations from each matrix. The PDF uses the full Pearson scale from `-1` to `1` and plots both ranked
matrices as full-page heatmaps. One additional graph per MoE node plots every finite correlation in sorted order,
with the previous-token and previous-layer series on the same axes. A second linear graph shows the 1,000 strongest
positive and 1,000 strongest negative correlations for both relations. The analyzer also exports
`<output-prefix>-selected-experts.csv` with one decode selection per row: the cumulative one-based decode iteration,
the zero-based layer-sorted MoE number, the local expert number, and its zero-based global position:

```bash
python tools/python/qmoe_expert_activation_correlations.py counter.log \
  --output-prefix counter-expert-activation-correlations
```

`qmoe_prompt_runner.py --until-eos` omits its usual artificial new-token limit and lets generation stop on EOS. The
model context length remains the unavoidable hard limit. Collecting counter traces with this runner requires
`--moe-cpu-offload-experts` to be positive; a run without offload produces no counter records.
The runner writes `[qmoe_prompt_runner] prefill 0` before submitting prompt tokens and
`[qmoe_prompt_runner] decode N` before each subsequent generation step. These markers identify the phase even for a
one-token prompt or a multi-token prefill selecting only eight distinct experts. Timeline and correlation analysis
exclude prefill; counter replay applies its updates but scores only decode. Replay retains both the prompt and
generation iteration so decode groups cannot span prompts. Decode analysis rejects older traces without this phase
metadata instead of guessing from the number of selected experts; regenerate those traces with the updated runner.

The CPU-offload strategy is selected with `session.moe_cpu_offload_strategy=adaptive|basic|layered` (default `adaptive`).
The QMoE-only `basic` strategy fills the chosen nodes' GPU expert slots, then places the remaining
experts on CPU, with an optionally split boundary node. Placement ignores saved popularities and never swaps,
regardless of the configured swap limit; usage counters remain available.
The QMoE-only `layered` strategy fills chosen layers up to `session.moe_cpu_offload_gpu_ratio` (default `0.8`),
then leaves the rest on CPU. Within each GPU-resident layer, including a partially filled boundary layer,
placement and swaps use popularity. The target is rounded down; if necessary, layers exceed it in the chosen order
only enough to preserve exactly `K` CPU experts.
Both strategies accept `session.moe_cpu_offload_layer_order=first|last|middle` (default `first`):
start from the beginning, end, or center of the graph-scope/topological layer list.
Middle alternates outward from the left center (`3,4,2,5,1,6,0,7` for 8 layers).
Graph execution order and saved popularity record numbering are unchanged. See
[CPU offload strategies](moe-cpu-offload/README.md#initial-placement-and-ownership) for ordering and restrictions.

After each successful CPU-offload `Run()` using `adaptive` or `layered`, INFO logging also emits one `moe_expert_swaps` JSON record. It reports
`in_flight` swaps and `blocked_candidates` per CUDA device, plus session-wide totals. Its `scheduled_swaps` array
identifies each exchange started by that run with the graph scope, node index, device, outgoing CUDA expert, incoming
CPU expert, and both pre-swap scores. A blocked candidate is currently eligible for a swap but was not started because
the per-device concurrency limit was reached. Candidates are recomputed after every run rather than retained in a
persistent queue. This record is not part of the counter-record logging budget.

CUDA QMoE also accepts `session.moe_cpu_offload_on_demand_threshold=N` (default `0`, disabled).
With `adaptive` or `layered`, an invocation missing at most `N` distinct experts synchronously exchanges them
with unselected GPU experts before computation. Prefill counts the union across all tokens and tiles; decode
uses the same rule. The selected union must fit the layer's existing GPU capacity, otherwise execution stays
on the existing CPU/GPU path without partial demand swaps. The exact CPU expert count is preserved.
This mechanism is independent of the background swap limit and reuses existing CUDA packing and CPU repacking.
For mixed CPU/GPU nodes, INFO logging adds `moe_expert_demand_swaps` records with request and node identity,
`missing_experts`, completed `swapped_experts`, and `all_required_on_cuda`. These are distinct from background
swap scheduling records and are not part of the counter-record logging budget.

A `KernelPilot` is allocated for each kernel at initialization; its `KernelPilotMoeExpertSelection` member counts repeated routing IDs only once.
Each counter is updated in place. The coefficient constraints keep it bounded by the larger of its initial value
and `1`, so no next-value buffer or overflow-validation pass is needed.

The counting path rejects overlapping `Run()` calls before execution begins. Counter updates therefore require no
per-counter synchronization, and one invocation cannot interleave selection resets or updates with another.
Registration, counter parameters, and file loading are frozen before execution starts.

Kernels call `OpKernelContext::GetKernelPilot()` to obtain their session-owned pilot, or `nullptr` when unavailable.
They reset its `Moe()` member for the invocation and collect local expert IDs directly into it.
After successful kernel execution, the executor calls `KernelPilotMoeExpertState::RecordUsage(kernel)` to commit the collected
usage. An invocation that does not access its collector does not replay a previous invocation's selection.
The internal C++ shared-provider bridge forwards only the getter from CUDA to the runtime.
`KernelPilot` has the same provider-independent definition on both sides; neither `KernelPilotMoeExpertState` nor in-tree graph
types cross the provider boundary.
`KernelPilotMoeExpertState::GetCounters(kernel, counters)` reads one kernel's counters, and
`KernelPilotMoeExpertState::GetExpertStats()` returns a flat `(kernel, local expert id, popularity)` list across all
registered kernels, for internal callers that only need live counter values rather than graph identity.
`SessionState::GetKernelPilot(kernel)` returns a kernel's `KernelPilot` directly, without exposing the
MoE-specific `KernelPilotMoeExpertState` type to that call path; `SessionState::GetMoeExpertState()` remains available for
initialization and inspection. There is no public C API, `OrtApi` entry, or Python
counter-retrieval API.

`KernelPilot` in `core/framework/kernel_pilot.h` is a generic, provider-independent per-kernel piloting object;
it carries no kernel-specific logic itself. `KernelPilotMoeExpertSelection`, defined in
`core/framework/kernel_pilot_moe_expert_selection.h`,
collects and deduplicates local expert IDs for one invocation, with reusable selection storage and no counter-update
logic. CPU kernels do not collect expert usage. The CUDA adapter, `KernelPilotMoeExpertSelectionCuda`, owns only the
device-transfer resources and feeds completed host snapshots into the `KernelPilotMoeExpertSelection` obtained from the context.
Both implement `IKernelPilotMoeExpertSelection`; the CUDA implementation delegates the shared selection API to the
session-owned selection after attaching it for an invocation.
`KernelPilotMoeExpertState` remains the sole owner of the global counters and their update logic.

For each MoE or QMoE invocation, the events are:

1. `Compute()` obtains the kernel's session-owned `KernelPilot` through `OpKernelContext::GetKernelPilot()`.
2. `BeginInvocation(expert_count)` clears the previous transient selection while retaining its allocated storage.
3. The routing stage computes local expert IDs. CUDA kernels enqueue a
   device-to-host copy and completion event through `KernelPilotMoeExpertSelectionCuda`.
4. CUDA submits the expert work, then `Consume()` waits only for the earlier copy event and calls `Collect()` while
   later expert work may remain in flight.
5. Tiled routing repeats the capture and consumption steps for every tile. The shared selection stores the union of
   expert IDs.
6. If the kernel returns successfully, `SequentialExecutor` calls `RecordKernelUsage()`, which forwards the kernel
   pointer to `KernelPilotMoeExpertState::RecordUsage()`. A failed kernel skips this commit.
7. `RecordUsage()` reads the selected IDs, applies the decay coefficient to all counters in that kernel's range, and
   adds the configured contribution to each selected expert. If logging is active, it then logs that counter update.
8. The next invocation calls `BeginInvocation()` again; counters persist, but the transient selection is reset.

CUDA counting queues a routing-ID copy into pinned host memory and a copy-completion event after top-k, before
expert computation. After submitting the expert kernels, the calling CPU thread waits only for that event and
processes usage while the GPU can continue expert computation; it does not synchronize the entire compute stream.
Fused QMoE routing keeps its fused prologue and queues the snapshot immediately after that prologue. Tiled QMoE
consumes each snapshot before reusing its host buffer and unions selected experts across all tiles before updating
the counters once for the invocation. Same-stream ordering protects tile-local device routing scratch.
Debug synchronization, `CUDA_LAUNCH_BLOCKING`, or optional tactic profiling can still serialize GPU work.
Pilots are owned by the session state and constructed once per kernel when expert counting or counter-update logging
is enabled. `KernelPilotMoeExpertSelection` resets the
current invocation's selected-expert set while reusing its storage. The CUDA adapter also reuses its pinned host
buffer and copy event. Buffer capacity grows only when needed for a larger invocation.

Counting, counter-update logging, and a non-empty popularity output path enable tracking at kernel construction.
With both flags disabled and no output path, the
kernel skips collector construction, collection calls, and statistics-only size calculations.
There are no statistics allocations, host transfers, or stream synchronizations on that path; only cached flag checks
remain. Enabling these diagnostics adds overhead.

## FP16/BF16 CUDA expert offload and adaptive placement

Set the session-wide CPU offload count with:

```text
session.moe_cpu_offload_experts=<non-negative integer>
```

CPU offload applies to built-in CUDA `MoE` nodes with FP16 or BF16 activations and constant
expert weights. Before session initialization returns, `KernelPilot` chooses the experts that remain resident in CUDA
memory. All-zero counters distribute slots round-robin across eligible nodes; initialized counters rank experts by
descending value with deterministic ties. The offload count is global and must not exceed the number of experts in
eligible CUDA nodes. The default is `0`, which disables CPU offloading.

With CPU offload enabled, eligible nodes use disjoint compact weight stores: CPU-resident experts have GEMM-layout
weights on CPU, while CUDA-resident experts have CUTLASS-layout weights on CUDA. Initialization retains ownership of
the original initializers without copying their complete contents into per-kernel CPU vectors. External expert weights
remain file-backed mappings until their node is initialized. The CUDA external-data loader gathers the selected CUDA
experts through its two reusable 64 MiB pinned buffers; only CPU-resident experts are materialized in CPU GEMM layout.
Each original initializer is released after its compact stores are ready, including for nodes with no CPU experts.
There is no permanent CPU copy of every CUDA-resident expert and no complete model-sized CPU staging copy.

During inference, CUDA experts run through the existing CUTLASS MoE path. CPU-resident FP16 experts use MLAS FP16
GEMMs; CPU-resident BF16 experts use FP32 GEMM-layout weights and MLAS FP32 GEMMs.
Their weighted outputs are combined on CUDA. Routing still records every selected expert, so counter updates continue
unchanged.

Offload requires prepacking to remain enabled and the input hidden dimension to be statically known. Connected
FC1/FC2 biases must be constant, just like the expert weights.
Expert-weight collection and validation follow actual outer-scope captures into subgraphs; same-named local inputs
and initializers are separate values and do not cause ancestor values to be validated or moved to CPU.

After each successful `Run()`, the runtime uses the updated counters to schedule adaptive exchanges within nodes that
have both CPU and CUDA experts. A node is eligible when its highest CPU counter exceeds
`(1 + epsilon)` times its lowest CUDA counter. Configure the relative margin and per-device concurrency limit with:

```text
session.moe_expert_swap_epsilon=<finite non-negative value> # default: 0.01
session.moe_max_in_flight_swaps_per_device=<non-negative integer> # default: 4
```

At most one background exchange per node is scheduled at each inference boundary. Transfers run asynchronously;
without on-demand swaps, an invocation uses the previously published placement if its exchange is still incomplete,
without waiting for it. The new placement is published only after both transfer directions complete.
Setting the concurrency limit to `0` disables background exchanges and keeps initial placement fixed unless on-demand
swaps are enabled. Each node retains its initial CUDA slot count; redistribution of slots between nodes is not implemented.

Separate FC3 weights, sparse mixer routing, dynamic expert weights, FP32, the CUDA plugin EP, CUDA graphs, and
minimal builds are not supported by dense MoE offload. INT4/INT8 QMoE also supports this option, subject to the
[QMoE offload restrictions](contrib_ops/cuda/moe_qmoe.md#31-int4int8-cpu-expert-offload).
See [MoE CPU offload](moe-cpu-offload/README.md) for exchange details and benchmark results.

## Popularity state file

Set an output path to write the final state when the session is destroyed:

```text
session.moe_expert_popularity_file=/path/to/popularity.json
```

This requires `session.moe_cpu_offload_experts` to be positive; the option does not enable counting by itself.
Without offload, the runtime warns and writes no popularity file. The benchmark wrapper exposes the output path as
`--popularity-output`. The UTF-8 JSON file contains the configured decay coefficient (`alpha`), swap threshold
(`epsilon`), and one record per expert:

```json
{
  "alpha": 0.9,
  "epsilon": 0.1,
  "experts": [
    {"moe": 0, "expert": 0, "popularity": 0.42},
    {"moe": 0, "expert": 1, "popularity": 0.17}
  ]
}
```

`moe` is a stable zero-based ordinal obtained by sorting resolved nodes by graph scope and node index. `expert` is the
zero-based expert index local to that MoE node. If the file cannot be written during session destruction, ONNX Runtime
logs an error.

Pass the resulting file back through `session.moe_expert_counter_state_file` or the benchmark's `--state` option to
restore the popularities, `alpha`, and `epsilon`. Values from the JSON replace explicitly configured alpha and epsilon.
The loaded alpha also replaces the coefficient cached by every CUDA pilot before the first inference.
Omitted experts start at zero. Out-of-range or duplicate expert records, invalid values, malformed JSON, and file-access
errors fail session initialization. Use the same model and graph-optimization configuration when loading a state file.
