# Adaptive CUDA expert offloading for Qwen 3.6 MoE

**Status:** Implementation in progress

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

The first offload implementation does not use counters to choose its initial placement. It randomly selects the
configured global number of experts to keep on CPU and keeps that placement fixed for the lifetime of the session.
This separates hybrid CPU/CUDA inference correctness from the adaptive policy. Counter-based placement changes are
added only in the second implementation step.

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

Before computing or scheduling redistribution, drain every pending per-node exchange: wait for each transfer-completion
event and publish its completed mapping. Redistribution therefore starts from a stable placement in which no transfer
can still publish an owner for a slot. Only after this drain may redistribution reassign slot ownership.
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
- submits counter updates and, once adaptive swaps are enabled, placement changes to the cache manager.

The implementation must verify that initializer prepacking and memory planning can retain canonical weights on CPU
without materializing every expert on CUDA. If the existing input-memory contract cannot support this without
regressing the regular CUDA path, an internal graph transformer may insert an experimental `MoEWithCPUOffload`
operator. That operator must reuse `MoE`/`QMoE` schema semantics and kernels and must not become part of the exported
model contract.

When `session.moe_cpu_offload_experts` is absent, CPU and CUDA `MoE`/`QMoE` behavior remains unchanged.

## State ownership and concurrency

The root `SessionState` owns a dedicated `KernelPilotMoeExpertState` shared with its subgraph session states. It contains:

- policy parameters;
- per-node expert counters;
- the global expert budget and per-node allocation targets.

CPU and CUDA kernels obtain their session-owned `KernelPilot` through `OpKernelContext::GetKernelPilot()`.
The generic pilot in `core/framework` carries no kernel-specific logic; `KernelPilotMoeExpertSelection`, defined in
`kernel_pilot_moe_expert_selection.h`, deduplicates selected expert IDs. CUDA-specific copies and events stay in a
provider-side adapter.
After successful kernel execution, the executor commits the collected usage through
`KernelPilotMoeExpertState::RecordUsage()`, which owns the
counter-update logic. These are internal C++ calls, not a public C API.
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

The session-global expert state and counters are already implemented. The remaining implementation is split into two
steps so that hybrid inference is validated before placement starts changing at runtime. Each pull request includes
the tests and documentation for its own scope.

### Step 1: random offload placement and hybrid inference

- Parse and validate the count-or-proportion offload target.
- Select exactly that global number of experts randomly for CPU offload during session initialization.
- Keep this initial placement immutable: this step has no swaps or end-of-inference redistribution.
- Retain canonical weights for every expert on CPU and copy only CUDA-resident experts to device slots.
- Dispatch resident experts on CUDA and offloaded experts through the shared CPU expert-compute path.
- Combine CPU and CUDA expert results without changing the exported `MoE` or `QMoE` model contract.
- Preserve the existing CUDA implementation when offloading is disabled.
- Test CPU-only, CUDA-only, and mixed expert execution, the global offload count, numerical agreement, bounded memory,
  repeated inference with a fixed placement, and unchanged disabled-path behavior.

### Step 2: adaptive expert swaps

- Use the session-global counters and optional initial counter state to rank experts.
- Apply the strict `cpu_max > (1 + epsilon) * cuda_min` rule and schedule at most one local exchange per node
  invocation.
- Manage CUDA slots, transfer streams, completion events, immutable per-invocation mappings, and atomic publication of
  completed swaps.
- Submit copies asynchronously after the current node finishes, overlap them with later model work, and require
  completion before that node executes again.
- Redistribute the global CUDA expert budget after inference, first draining pending local exchanges and then
  maximizing the number of completely CUDA-resident nodes.
- Test the epsilon boundary, safe slot reuse, asynchronous publication, global budget preservation, delayed exchanges
  followed by redistribution, counter-based placement, and explicit transfer failures.

After these two implementation steps, the remaining work is end-to-end measurement. Run reproducible CPU-only,
CUDA-only, and hybrid evaluations with identical models, prompts, and generation settings; sweep offload targets and
policy parameters; and record the latency, throughput, memory, placement, and transfer metrics listed below. Commit
the evaluation scripts, aggregate results, and documented commands while keeping oversized raw traces outside the
repository.

## Validation

Tests cover:

- count and proportion parsing, including exact rounding and invalid values;
- random selection of exactly the configured global number of offloaded experts;
- immutable placement across repeated inference in the first implementation step;
- CPU-only, CUDA-only, and mixed expert execution with numerical agreement;
- absent, complete, partial, malformed, and inconsistent initial state;
- exponential updates for used and unused experts;
- deterministic counter ties;
- the strict `cpu_max > (1 + epsilon) * cuda_min` boundary;
- no exchange when a node is entirely on CPU or entirely on CUDA;
- at most one exchange per node invocation;
- no copy before node completion;
- safe slot reuse after CUDA completion;
- asynchronous copy submission and required completion before the next invocation;
- atomic mapping publication;
- draining pending per-node exchanges before redistribution can reassign their slots;
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
