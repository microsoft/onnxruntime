# Workspace-Aware Memory Optimization

> **Goal:** Make constrained-environment partitioning account for kernel workspace before
> assigning nodes, then reuse activation memory for that workspace at runtime.

## One Contract, Two Estimation Levels

```text
Maximum-shape hints (#31613, merged)
             |
             v
Level 1: pre-kernel estimate --------> GetCapability() memory budget
             |                              (#31962)
             v
Level 2: post-kernel declaration ----> verify Level 1 reservation
             |                              (#32189)
             v
Trace workspace lifetime ------------> pack with dead activations
             |                              (#32071)
             v
Runtime bounds check ----------------> planned pointer or dynamic fallback
```

| | Level 1 | Level 2 |
|---|---|---|
| **Question** | Can this partition fit? | What does this kernel require? |
| **When** | Before kernel creation | After kernel creation and prepacking |
| **Information** | Node, planning shapes, EP/device | Kernel and selected algorithm |
| **Use** | Budgeting and EP assignment | Verification and preallocation |
| **Safety** | Profile/fallback if unresolved | Dynamic fallback if unusable or exceeded |

## Why This Matters

- Existing fallback accounting can be too conservative or incomplete.
- Runtime scratch allocations add allocator traffic and cannot share known-dead activation
  storage through the memory pattern.
- Level 1 prevents workspace from being omitted during partitioning.
- Level 2 catches underestimated reservations and provides a precise allocation contract.
- Runtime tracing captures actual lifetimes, enabling safe workspace/activation overlap.

## Pilot Scope and Evidence

**Pilot:** in-tree CUDA `MatMulNBits`, sequential execution, one workspace slot, dynamic
fallback retained.

Local RTX 5090 Laptop measurements for synthetic 64-token prefill:

| Model | Arena reservation | Arena allocation calls | Average latency | WDDM peak |
|---|---:|---:|---:|---:|
| Qwen 2.5 1.5B | -250,112 B | -30.0% | -24.8% | No change |
| Hy-MT2 1.8B | +7,680 B | -43.2% | -14.4% | No change |

The pilot demonstrates activation/workspace reuse and substantially fewer arena calls.
Memory savings remain shape-dependent, and the latency measurements cover prefill only,
not batch-1 token-by-token decode.

## Team Decisions and Next Steps

1. Confirm Level 1 as the partitioning contract and Level 2 as its post-assignment
   verification/allocation contract.
2. Decide whether Level-2 excess should warn, fail in strict mode, or trigger future
   repartitioning.
3. Expand estimator/declaration coverage to high-workspace kernels.
4. Generalize lifetime planning to multiple slots, parallel execution, streams, and plugin
   execution providers.

**Tracking:** [#29775](https://github.com/microsoft/onnxruntime/issues/29775) |
**Design:** [Workspace Estimation and Preallocation](workspace_estimation_and_preallocation.md) |
**PR stack:** [#31962](https://github.com/microsoft/onnxruntime/pull/31962) ->
[#32189](https://github.com/microsoft/onnxruntime/pull/32189) ->
[#32071](https://github.com/microsoft/onnxruntime/pull/32071)
