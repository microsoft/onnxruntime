# Session-global MoE expert counters

Enable expert usage counters with the session configuration entry:

```text
session.enable_moe_expert_counting=1
```

The default is `0`. Counting alone does not emit logs, enable profiling, change expert placement, or swap weights.
`session.enable_moe_expert_statistics=1` enables the same counter updates and emits one structured
`moe_expert_counters` log after each successful MoE kernel invocation, subject to the per-Run logging limit below.
Configure the update with:

```text
session.moe_expert_counter_alpha=<finite non-negative value> # default: 0.9
session.moe_expert_counter_beta=<finite non-negative value>  # default: 0.1
```

The coefficients must satisfy `alpha + beta <= 1`; zero is allowed for either coefficient.

It supports the CPU and built-in CUDA `MoE` and `QMoE` kernels. Minimal builds, the CUDA plugin EP, and CUDA graph
capture are not supported when either counting or counter-update logging is enabled.
Minimal builds omit the counter state and its initialization/run bookkeeping, and reject counter configuration
except for explicitly disabling counting.

Each node invocation updates every expert counter belonging to that node:

```text
c(t+1) = alpha * c(t) + beta * (1 if used otherwise 0)
```

Selecting an expert for several token rows still contributes one `beta`. Unused experts still receive the decay term.
Counts persist across `Run()` calls. Only successful kernel invocations commit their collected usage.
A failed run can retain updates from earlier successful kernels, including CUDA nodes whose
expert computations were still in flight when their counters were updated. Updates are not transactional across an
entire model invocation.

The root `SessionState` owns a `KernelPilotMoeExpertState` shared with its subgraph states. Sessions never share counters.
While counter-update logging is active, the state also stores the current request ID and logger. Because expert
selections and counters are session-owned mutable state, a concurrent Run is rejected whenever counting or
counter-update logging is enabled.
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
Logging is limited to 1024 counter records per `Run()`, shared across all nodes and subgraphs.
On the first omitted record, a single WARNING `moe_expert_counters_truncated` JSON marker reports
`request_id` and `max_records`. Later records are omitted without formatting their JSON.
Counter updates continue unchanged after the limit, and the logging budget resets on the next Run.
The Python analyzer rejects any trace containing this marker, even with an otherwise valid completion footer.
When INFO output is disabled, counter records are neither formatted nor charged against the logging budget.
Only logging-budget reservation uses an atomic; counting-only updates remain unsynchronized across disjoint kernel ranges.
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
logic. CPU kernels feed it
their host routing IDs directly through `pilot->Moe()`. The CUDA adapter, `KernelPilotMoeExpertSelectionCuda`, owns only the
device-transfer resources and feeds completed host snapshots into the `KernelPilotMoeExpertSelection` obtained from the context.
Both implement `IKernelPilotMoeExpertSelection`; the CUDA implementation delegates the shared selection API to the
session-owned selection after attaching it for an invocation.
`KernelPilotMoeExpertState` remains the sole owner of the global counters and their update logic.

For each MoE or QMoE invocation, the events are:

1. `Compute()` obtains the kernel's session-owned `KernelPilot` through `OpKernelContext::GetKernelPilot()`.
2. `BeginInvocation(expert_count)` clears the previous transient selection while retaining its allocated storage.
3. The routing stage computes local expert IDs. CPU kernels call `Collect()` with host IDs. CUDA kernels enqueue a
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

Both counting and counter-update logging options are cached at kernel construction. With their default values (`0`), the
kernel skips collector construction, collection calls, and statistics-only size calculations.
There are no statistics allocations, host transfers, or stream synchronizations on that path; only cached flag checks
remain. Enabling these diagnostics adds overhead.

## Initial counter file

Optionally set:

```text
session.moe_expert_counter_state_file=/path/to/counters.txt
```

This requires counting or counter-update logging to be enabled. The UTF-8 text file begins with
`moe_expert_state 1`, followed by whitespace-separated records:

```text
moe_expert_state 1
"main" 0 MoE 2 12
"main" 1 QMoE 0 3.5
"main/4/11:then_branch" 0 MoE 1 7
```

Each record contains a quoted graph scope, node index, operator type, expert ID, and initial counter value. The root
scope is `main`. A subgraph appends `/<parent-node-index>/<attribute-name-length>:<attribute-name>` to its parent's
scope. This distinguishes nodes with the same index in different subgraphs. Node indices refer to the resolved,
optimized graph, not necessarily the original serialized graph.

Values must be finite and non-negative. Omitted experts start at zero. Without a file, all experts start at zero.
Unknown nodes, mismatched operator types, out-of-range expert IDs, duplicate records, malformed files, and file-access
errors fail session initialization. Use the same model and graph-optimization configuration when preparing and loading
a counter file.
