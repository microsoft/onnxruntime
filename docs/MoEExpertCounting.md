# Session-global MoE expert counters

Enable expert usage counters with the session configuration entry:

```text
session.enable_moe_expert_counting=1
```

The default is `0`. Counting is independent of `session.enable_moe_expert_statistics` and does not emit routing logs,
enable profiling, change expert placement, or swap weights. Configure the update with:

```text
session.moe_expert_counter_alpha=<finite non-negative value> # default: 0.9
session.moe_expert_counter_beta=<finite non-negative value>  # default: 0.1
```

The coefficients must satisfy `alpha + beta <= 1`; zero is allowed for either coefficient.

It supports the CPU and built-in CUDA `MoE` and `QMoE` kernels. Minimal builds, the CUDA plugin EP, and CUDA graph
capture are not supported with counting enabled.
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

The root `SessionState` owns a `MoeExpertState` shared with its subgraph states. Sessions never share counters.
Registration uses graph scope and resolved node index, with one counter per expert. The router's expert dimension must
be statically known when the session is initialized. The global expert count is the sum of those per-node dimensions;
for `N` equally sized MoE nodes with `E` experts, it is `N * E`.

After kernel creation, initialization builds an immutable dictionary
`(OpKernel pointer, local expert ID) -> global expert index`, plus the contiguous expert range for each kernel.
Counters are ordinary `double` values in a session-wide array. `RecordUsage()` uses the kernel pointer and this
dictionary directly, without a mutex, atomic counters, graph-name lookup, or per-invocation allocation.
A `KernelUsage` collector is allocated for each kernel at initialization to count repeated routing IDs only once.
Each counter is updated in place. The coefficient constraints keep it bounded by the larger of its initial value
and `1`, so no next-value buffer or overflow-validation pass is needed.

Counting requires non-overlapping `Run()` calls on a session. An entry guard rejects overlapping calls explicitly;
there is no locking in the kernel update path. Different kernels may update their disjoint ranges concurrently, but
one kernel must not execute twice simultaneously. Global snapshots must be read while no run is active. Registration,
counter parameters, and file loading are frozen before execution starts.

Kernels call `OpKernelContext::GetKernelUsage()` to obtain their session-owned collector, or `nullptr` when collection
is unavailable. They reset it for the invocation and collect local expert IDs directly into it.
After successful kernel execution, the executor calls `MoeExpertState::RecordUsage(kernel)` to commit the collected
usage. An invocation that does not access its collector does not replay a previous invocation's selection.
The internal C++ shared-provider bridge forwards only the getter from CUDA to the runtime.
`KernelUsage` has the same provider-independent definition on both sides; neither `MoeExpertState` nor in-tree graph
types cross the provider boundary.
`MoeExpertState::GetCounters(kernel, counters)` reads one kernel's counters.
`SessionState::GetMoeExpertState()->GetSnapshot()` provides an internally accessible snapshot of all registered nodes.
There is no public C API, `OrtApi` entry, or Python counter-retrieval API.

`KernelUsage` in `core/framework/kernel_usage.h` collects and deduplicates local expert IDs for one invocation.
It is a concrete, provider-independent collector shared by CPU and CUDA, with reusable selection storage and no
counter-update logic. CPU kernels feed it their host routing IDs directly. The CUDA adapter, `CudaRoutingSnapshot`,
owns only the device-transfer resources and feeds completed host snapshots into the collector obtained from the context.
`MoeExpertState` remains the sole owner of the global counters and their update logic.

CUDA counting queues a routing-ID copy into pinned host memory and a copy-completion event after top-k, before
expert computation. After submitting the expert kernels, the calling CPU thread waits only for that event and
processes usage while the GPU can continue expert computation; it does not synchronize the entire compute stream.
Fused QMoE routing keeps its fused prologue and queues the snapshot immediately after that prologue. Tiled QMoE
consumes each snapshot before reusing its host buffer and unions selected experts across all tiles before updating
the counters once for the invocation. Same-stream ordering protects tile-local device routing scratch.
Debug synchronization, `CUDA_LAUNCH_BLOCKING`, or optional tactic profiling can still serialize GPU work.
Collectors are owned by the session state and constructed once per kernel only when
`session.enable_moe_expert_counting=1`. `KernelUsage` resets the
current invocation's selected-expert set while reusing its storage. The CUDA adapter also reuses its pinned host
buffer and copy event. Buffer capacity grows only when needed for a larger invocation.

Both counting and routing-logging options are cached at kernel construction. With their default values (`0`), the
kernel skips collector and instrumentation-context lookups, collector construction, collection calls, and
statistics-only size calculations.
There are no statistics allocations, host transfers, or stream synchronizations on that path; only cached flag checks
remain. Enabling these diagnostics adds overhead.

## Initial counter file

Optionally set:

```text
session.moe_expert_counter_state_file=/path/to/counters.txt
```

This requires counting to be enabled. The UTF-8 text file begins with `moe_expert_state 1`, followed by whitespace-separated
records:

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
