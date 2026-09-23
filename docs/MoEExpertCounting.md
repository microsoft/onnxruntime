# Session-global MoE expert counters

Enable expert usage counters with the session configuration entry:

```text
session.enable_moe_expert_counting=1
```

The default is `0`. Counting is independent of `session.enable_moe_expert_statistics` and does not emit routing logs,
enable profiling, change expert placement, or swap weights. Configure the update with:

```text
session.moe_expert_counter_alpha=<finite value in [0, 1]>  # default: 1
session.moe_expert_counter_beta=<finite non-negative value> # default: 1
```

It supports the CPU and built-in CUDA `MoE` and `QMoE` kernels. Minimal builds, the CUDA plugin EP, and CUDA graph
capture are not supported with counting enabled.

Each successful node invocation updates every expert counter belonging to that node:

```text
c(t+1) = alpha * c(t) + beta * (1 if used otherwise 0)
```

Selecting an expert for several token rows still contributes one `beta`. Unused experts still receive the decay term.
Counts persist across `Run()` calls. A failed run can retain updates from nodes that completed before the failure;
updates are not transactional across an entire model invocation.

The root `SessionState` owns a `MoeExpertState` shared with its subgraph states. Sessions never share counters.
Registration uses graph scope and resolved node index, with one counter per expert. The router's expert dimension must
be statically known when the session is initialized. The global expert count is the sum of those per-node dimensions;
for `N` equally sized MoE nodes with `E` experts, it is `N * E`. Updates and snapshot reads are protected by a mutex.

Internal kernels access their own counters through `OpKernelContext::HasMoeExpertState()`,
`RecordMoeExpertUsage()`, and `GetMoeExpertCounters()`. The shared-provider bridge exposes the same interface to CUDA.
`SessionState::GetMoeExpertState()->GetSnapshot()` provides an internally accessible snapshot of all registered nodes;
this change does not add a public C or Python counter-retrieval API.

CUDA counting copies routing IDs to the host and synchronizes the compute stream before consuming them. Tiled QMoE
unions selected experts across all tiles before updating the counters once for the invocation. This opt-in diagnostic
mode adds synchronization overhead; disabled counting does not allocate counter buffers or synchronize.

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
