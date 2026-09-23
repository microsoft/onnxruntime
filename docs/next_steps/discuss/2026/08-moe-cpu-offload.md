# Adaptive CUDA expert offloading for Qwen 3.6 MoE

**Status:** Implementation planned

**Date:** 2026-09

## Objective

Implement adaptive expert placement for Qwen 3.6 and other Mixture-of-Experts (MoE) models whose expert weights do not
all fit in GPU memory.

CPU memory keeps the canonical copy of every expert. CUDA holds a configurable subset of expert copies. Each `MoE` and
`QMoE` node maintains one exponentially decayed counter per expert, uses those counters to rank experts, and updates
its CUDA placement asynchronously.

The placement policy has two levels:

- After a `MoE` or `QMoE` invocation, exchange a hot CPU expert with a cold CUDA expert when the counter difference
  exceeds a configurable threshold.
- After a complete model inference, redistribute the global CUDA expert budget across nodes while maximizing the
  number of nodes that can run entirely on CUDA.

Training, router changes, expert-weight quantization, and multiple CUDA devices are outside this implementation.

## Global offload configuration

The four numerical policy parameters are exposed as session configuration entries:

| Session option | Parameter | Meaning and valid range |
|---|---|---|
| `session.moe_cpu_offload_experts` | Offload target | Global integer expert count (`>= 1`) or proportion (`0 < value < 1`). |
| `session.moe_expert_counter_alpha` | `alpha` | Counter decay coefficient, finite and `>= 0`; default `0.9`. |
| `session.moe_expert_counter_beta` | `beta` | Increment for a used expert, finite and `>= 0`; default `0.1`. |
| `session.moe_expert_swap_epsilon` | `epsilon` | Relative swap margin, finite and `>= 0`. |

The optional `session.moe_expert_counter_state_file` path is configured separately from these four numerical parameters.

The offload target has the following meaning:

- An integer greater than or equal to `1` is the total number of experts to offload to CPU.
- A value strictly between `0` and `1` is the proportion of all experts to offload to CPU. The concrete expert count is
  `ceil(value * total_expert_count)`.
- Zero, negative values, non-integral values greater than `1`, non-finite values, and counts larger than the total
  number of experts are invalid.
- When the option is absent, expert offloading is disabled.

The global CUDA budget is:

```text
cuda_expert_count = total_expert_count - cpu_offload_expert_count
```

One CUDA slot contains all weights required to execute one expert. CUDA slots contain copies only; moving an expert
into or out of CUDA never removes or modifies its canonical CPU weights.

## Expert counters

Each `MoE` and `QMoE` node owns one counter for each of its experts. After the node executes, every counter is updated
once:

```text
count_{t+1}(e) = alpha * count_t(e) + beta * (1 if expert e was used, otherwise 0)
```

The used term is binary for one invocation. Selecting the same expert for several rows still contributes `1`, not the
number of routed rows.

The policy exposes `alpha`, `beta`, and `epsilon` as validated non-negative parameters. The counter coefficients must
satisfy `alpha + beta <= 1`; zero is allowed for either coefficient. Counters are updated in place and remain bounded
by the larger of their initial value and `1`. Their defaults and constraints are covered by option-parsing tests.

Expert identity is `(graph_scope, node_index, node_type, expert_id)`. Ranking is by descending counter. Ties are resolved
by `expert_id`, then graph scope and node index, so placement is deterministic.

## Initial counter state

An optional session setting names a UTF-8 text file containing initial counter values:

```text
session.moe_expert_counter_state_file=<path>
```

The file starts with a format-version line and contains one line per
`(graph_scope, node_index, node_type, expert_id, counter_value)` record. Loading validates:

- the format version;
- graph scope, node identity, and operator type;
- expert index bounds;
- uniqueness of every expert record;
- finite, non-negative counter values.

Experts omitted from the file start at zero. When the setting is absent, all counters start at zero.

If all counters are zero, the global CUDA budget is distributed as uniformly as possible across `MoE` and `QMoE`
nodes. Any remainder is assigned in node-index order. Within each node, the lowest expert IDs are selected first.

## Per-node placement update

One invocation uses an immutable expert-to-slot mapping:

- CUDA-resident experts execute on CUDA.
- Non-resident experts execute from their permanent CPU weights.
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

At most one exchange is scheduled per node invocation. Counter ties use expert ID order.

The exchange starts immediately after the node finishes:

1. Record completion of all CUDA work that references the current slot.
2. Make the transfer stream wait for that completion event.
3. Copy the selected CPU expert into the selected CUDA slot asynchronously.
4. Record a transfer-completion event.
5. Publish the new mapping atomically after the copy completes.

The current invocation never waits for the exchange. Before the node's next invocation, the cache manager waits for
any pending exchange to finish. The next invocation must use the new complete mapping; it must not observe a partially
copied slot or continue with the replaced mapping.

```text
token t, node L
    -> execute with immutable placement P
    -> update counters
    -> evaluate cpu_max > (1 + epsilon) * cuda_min
    -> enqueue one qualifying exchange asynchronously
    -> continue the model

token t+1, before node L
    -> finish the pending exchange if necessary
    -> publish placement P+1
    -> execute with P+1
```

## End-of-inference redistribution

After each complete model inference, recompute how many CUDA slots belong to each `MoE` and `QMoE` node while preserving
the global CUDA budget.

The allocation objective is lexicographic:

1. Maximize the number of nodes whose complete expert set is CUDA-resident.
2. Among allocations with the same number of complete CUDA nodes, maximize the sum of counters retained on CUDA.
3. Break remaining ties by node index and expert ID.

Within each node, keep the experts with the highest counters. Redistribution may transfer slot ownership between
nodes, whereas a per-node exchange changes the expert stored in a slot without changing that node's slot count.

Redistribution copies are asynchronous. Every affected node must finish its pending redistribution before its next
invocation. Slot metadata is published only after all weights for that slot are ready.

## Operator integration

The implementation extends `com.microsoft::MoE` and `com.microsoft::QMoE` without changing the exported ONNX model
contract.

The graph node remains assigned to the CUDA execution provider. Its CUDA kernel:

- owns or accesses the runtime expert cache;
- keeps canonical expert weights in CPU memory;
- dispatches resident experts to CUDA;
- invokes shared CPU expert-compute helpers for non-resident experts;
- submits counter updates and placement changes to the cache manager.

The implementation must verify that initializer prepacking and memory planning can retain canonical weights on CPU
without materializing every expert on CUDA. If the existing input-memory contract cannot support this without
regressing the regular CUDA path, an internal graph transformer may insert an experimental `MoEWithCPUOffload`
operator. That operator must reuse `MoE`/`QMoE` schema semantics and kernels and must not become part of the exported
model contract.

When `session.moe_cpu_offload_experts` is absent, CPU and CUDA `MoE`/`QMoE` behavior remains unchanged.

## State ownership and concurrency

The root `SessionState` owns a dedicated `MoeExpertState` shared with its subgraph session states. It contains:

- policy parameters;
- per-node expert counters;
- the global expert budget and per-node allocation targets.

Kernels report selected expert IDs through `OpKernelContext::RecordMoeExpertUsage()`. The context forwards the kernel
identity and IDs to `MoeExpertState::RecordUsage()`, which owns the update logic. This is an internal C++ call, not a
public C API.
Expert identity includes graph scope as well as node and expert IDs, so nodes in different
subgraphs cannot collide. The state persists across `Run()` calls and is isolated from other sessions.

The CUDA cache manager owns device-specific resources and execution state:

- CUDA slots and current immutable mappings;
- pending exchanges and redistribution transfers;
- CUDA completion events.

Initialization builds an immutable dictionary from `(OpKernel pointer, local expert ID)` to a global expert index.
Counters are ordinary values; their update path does not acquire a mutex. Counting and adaptive placement require
non-overlapping `Run()` calls on a session, enforced at run entry. Distinct kernels may update disjoint expert ranges,
but a kernel cannot execute twice simultaneously. Global snapshots are read between runs.
Each invocation captures a stable placement mapping, and a slot cannot be overwritten until all work using its
previous contents has completed.

Errors are explicit. Invalid configuration, invalid initial state, allocation failures, copy failures, and event
failures fail session initialization or execution rather than silently disabling offload or retaining stale placement.

## Pull request plan

Each PR includes the tests and documentation for its own scope.

### PR 1: runtime cache manager

Provide independently tested cache and policy components without activating offload in the model's kernels.

- Parse and validate the count-or-proportion offload target.
- Parse and validate `alpha`, `beta`, and `epsilon`.
- Load the optional counter-state text file and initialize missing counters to zero.
- Build deterministic all-zero placement.
- Implement counter-update helpers and immutable per-invocation mappings.
- Implement threshold-based per-node exchanges.
- Implement end-of-inference global redistribution.
- Manage CUDA slots, streams, events, and atomic mapping publication.
- Add unit tests for configuration, initial state, ranking, counter updates, exchanges, and redistribution.

### PR 2: session-global expert state and counters

Depends on PR 1.

- Add `MoeExpertState` owned by the root `SessionState` and shared with subgraph session states.
- Register one counter for each expert of each `MoE` and `QMoE`, with graph-scoped node identity.
- Build the immutable kernel-pointer/expert-ID dictionary after kernel creation and before any run.
- Keep counter updates and reads in `MoeExpertState`, keyed by kernel pointer. Expose a kernel-context recording method
  and forward it through the internal C++ provider bridge; do not add per-kernel usage objects or extend the public C API.
- Load optional initial values from `session.moe_expert_counter_state_file`; initialize unspecified counters to zero.
- Wire CPU and CUDA MoE/QMoE routing results into per-invocation updates:

  ```text
  c(t+1) = alpha * c(t) + beta * (1 if used otherwise 0)
  ```

- Apply decay to every expert and add `beta` once per selected expert, even if several rows select it.
- For CUDA, enqueue a pinned routing snapshot and completion event before expert computation, then process usage
  on the CPU after launching the expert kernels. Wait only for the snapshot, not the entire compute stream.
- Preserve counters across `Run()` calls, isolate sessions, and reject overlapping runs. Use ordinary counters with
  no mutex in `RecordUsage()`, and read global snapshots only between runs.
- Keep counting opt-in and preserve model outputs and execution placement. Do not enable ranking, placement
  strategy, offload, swaps, or redistribution in this PR.
- Test zero and file initialization, used/unused experts, repeated selection within one invocation, persistence across
  runs, session isolation, subgraph identity, disjoint kernel updates, rejected overlapping runs, and CPU/CUDA agreement.

### PR 3: adaptive placement and MoE/QMoE offload

Depends on PRs 1 and 2.

Connect the session-global expert state and CUDA cache manager to every participating CUDA `MoE` and `QMoE` node using
the following strategy.

**Configuration and initial placement**

- Wire the four session options `session.moe_cpu_offload_experts`, `session.moe_expert_counter_alpha`,
  `session.moe_expert_counter_beta`, and `session.moe_expert_swap_epsilon` to the cache manager.
- Apply `session.moe_cpu_offload_experts` globally across all participating nodes, not separately to each node. Values
  greater than `1` specify an integer expert count; values strictly between `0` and `1` specify a proportion. The value
  `1` specifies one expert. Derive the global CUDA budget from the complementary expert count.
- Use the session-global per-expert counters, initialized from the optional text file or to zero.
- Rank experts by descending counter and select the highest-ranked experts up to the global CUDA budget. Count the
  selected experts belonging to each node to determine that node's initial CUDA allocation. Resolve ties
  deterministically. If all counters are zero, distribute CUDA slots uniformly across nodes instead.
- Retain canonical weights for every expert on CPU and copy only the selected experts to CUDA.

**Each MoE/QMoE invocation**

- Complete any pending exchange or redistribution for the node before it executes, then capture its immutable
  placement mapping.
- Dispatch resident experts on CUDA and non-resident experts on CPU, sharing routing validation and CPU expert
  computation with the regular kernels.
- After execution, update every expert counter for that node, including unused experts:

  ```text
  c(t+1) = alpha * c(t) + beta * (1 if used otherwise 0)
  ```

- Find the highest counter among the node's CPU experts and the lowest counter among its CUDA experts. If
  `cpu_max > (1 + epsilon) * cuda_min`, exchange those two experts. Nodes entirely on one device have no local exchange.
- Enqueue the exchange asynchronously immediately after node execution completes, so the weight copy can overlap
  subsequent model computation. Publish the new mapping only when the copy completes. The exchange must finish before
  that node's next invocation; wait for its completion event if necessary.

**After the complete model inference**

- Reevaluate the number of experts kept on CUDA for each `MoE` and `QMoE` while preserving the global offload target.
- Prioritize the largest possible number of nodes whose experts all reside on CUDA, then maximize retained counter
  mass. Within each node, select experts by descending counter.
- Schedule the resulting transfers safely with any pending per-node exchanges. Every affected node must finish its
  placement update before its next execution.

**Integration tests**

- Exercise global count/proportion settings, text-file initialization, all-zero uniform placement, and counter-based
  allocation across multiple MoE/QMoE nodes.
- Verify the exponential update for used and unused experts, the strict epsilon threshold, asynchronous exchange
  timing, and required completion before the next invocation.
- Verify global budget preservation and redistribution toward complete CUDA-resident nodes after inference.
- Cover rejected overlapping runs, numerical agreement, bounded memory, and unchanged behavior when offloading is disabled.

### PR 4: end-to-end evaluation

Depends on PR 3.

- Run reproducible CPU-only, CUDA-only, and hybrid evaluations with the same model, prompts, and generation settings.
- Sweep offload targets and policy parameters, including zero-initialized and file-initialized counters.
- Record the latency, throughput, memory, placement, and transfer metrics listed below.
- Commit evaluation scripts, aggregate results, and documented commands; keep oversized raw traces outside the repository.

## Validation

Tests cover:

- count and proportion parsing, including exact rounding and invalid values;
- absent, complete, partial, malformed, and inconsistent initial state;
- all-zero uniform placement and deterministic remainder assignment;
- exponential updates for used and unused experts;
- deterministic counter ties;
- the strict `cpu_max > (1 + epsilon) * cuda_min` boundary;
- no exchange when a node is entirely on CPU or entirely on CUDA;
- at most one exchange per node invocation;
- no copy before node completion;
- safe slot reuse after CUDA completion;
- asynchronous copy submission and required completion before the next invocation;
- atomic mapping publication;
- global budget preservation during redistribution;
- maximizing complete CUDA-resident nodes before retained counter mass;
- immutable per-invocation snapshots and rejection of overlapping runs;
- unchanged behavior when offloading is disabled;
- numerical agreement for CPU-only, CUDA-only, and mixed expert execution;
- bounded CPU and CUDA memory for partial and full offload targets.

## Performance evaluation

Measure:

- time to first token and inter-token latency;
- token throughput;
- peak CPU and CUDA memory;
- CUDA hit rate per node and overall;
- CPU fallback count;
- exchanges and global redistributions;
- host-to-device bytes and transfer count;
- overlap between transfers and model execution;
- time spent waiting for unfinished transfers;
- number of `MoE`/`QMoE` nodes running entirely on CUDA;
- output agreement with CPU-only and CUDA-only execution.

Kernel-only timing is reported separately from end-to-end latency and throughput.
