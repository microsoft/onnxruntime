# CUDA Attention Workspace Estimation Roadmap

## Goal and scope

This work provides operator-specific workspace estimation for the CUDA Attention family. It defines how an
Attention kernel derives a workspace recipe from a plain problem description and a selected backend.

The generic memory-estimation framework is a separate work track. PA and PMHA now use its Level-1 and Level-2
declaration points, but partition budgeting, memory planning, and planned-root or preallocation integration remain
outside this operator-specific work.

Related design and rollout:

- [Constrained-environment memory roadmap](https://github.com/microsoft/onnxruntime/issues/29775)
- [Generic workspace framework and future allocation modes](future_directions_constrained_env.md#phase-a-workspace-pre-declaration-declareworkspacerequirements)
- [Optional-aware Level-2 input-shape contract](https://github.com/microsoft/onnxruntime/pull/32312)
- [Activation memory-pattern planner](https://github.com/microsoft/onnxruntime/pull/32071)
- [PA/PMHA Level-1 and Level-2 adapters](https://github.com/microsoft/onnxruntime/pull/32321)

The Attention work owns:

- graph-free problem descriptions and checked workspace recipes;
- backend- and layout-specific workspace formulas;
- runtime allocation and view parity;
- route and boundary tests; and
- thin Attention-specific adapters after the generic framework API stabilizes.

## Runtime dispatch and AOT estimation

CUDA EP assignment happens during session initialization. Backend dispatch inside an assigned CUDA Attention kernel
can happen later, using the concrete inputs available to `Compute()`. These are distinct decisions.

Runtime sizing can be exact because the kernel first selects a backend and then requests that backend's recipe:

```text
concrete inputs -> runtime dispatch -> selected backend -> exact workspace recipe
```

Ahead-of-time (AOT) estimation cannot assume that the same selection is known. Backend reachability can depend on
optional inputs, build and runtime options, cache state, runner availability, device properties, and concrete dynamic
sequence lengths. When the exact backend cannot be proven, AOT estimation must enumerate the reachable backend recipes
and take a safe upper bound:

```text
shapes and bounds -> reachable backends -> recipe per backend -> maximum workspace
```

An estimator must not copy the runtime dispatch cascade or assume that dispatch is monotonic with shape. For example,
a larger shape may use a fused backend with small workspace while a nearby smaller shape falls back to an unfused
backend with an `S^2` attention buffer. Head eligibility is also non-monotonic: a componentwise upper bound can exceed
a backend's supported head range while a smaller runtime head remains supported, and equal-head routes can be reachable
under unequal Q/K and V bounds. Each route reachable at a positive runtime geometry within the bounds must therefore be
sized using the original componentwise maximum geometry, whose current workspace terms are monotonic. PA's
`qkv_hidden_sizes` values are immutable node attributes, not shape bounds, so their derived head sizes retain exact
runtime eligibility checks. Only PA geometry derived without that attribute and PMHA geometry supplied through
`WorkspaceInputShape` use bounded head reachability.

An AOT result should therefore be classified as:

- **Exact:** the backend and all governing dimensions are proven.
- **Safe bound:** the maximum workspace across all reachable backend recipes.
- **Unavailable:** a required shape, optional-input, capability, or recipe contract is not available.

## Recipe architecture

The reusable implementation follows this boundary:

```text
operator shapes and attributes
          |
          v
plain Attention problem
          |
          v
checked backend/layout recipe
          |
          v
runtime allocation and workspace views
```

The plain problem and recipe must not depend on `Node`, `NodeArg`, `GraphViewer`, `TensorShape`, or CUDA runtime types.
A future framework adapter may translate framework inputs into the plain problem, but it must not duplicate sizing or
validation arithmetic.

Recipes must use checked arithmetic, enforce relevant CUDA ABI and grid limits, and prove that every derived view is
contained in the allocated workspace.

## PR1: packed Attention recipes

[microsoft/onnxruntime#32283](https://github.com/microsoft/onnxruntime/pull/32283) is the first implementation in the
roadmap tracked by [microsoft/onnxruntime#29775](https://github.com/microsoft/onnxruntime/issues/29775). Its scope is
`PackedAttention` (PA) and `PackedMultiHeadAttention` (PMHA), plus a shared correction that widens CUTLASS MEA
attention-bias stride arithmetic for all MEA consumers.

PR1 establishes a single sizing and layout source of truth while preserving legacy workspace byte totals, allocation
counts, and allocation lifetimes. It does not change Attention backend selection. The shared stride correction can
change MEA's internal aligned-versus-unaligned kernel variant only in cases where the previous int32 calculation
overflowed before assignment to an int64 stride.

### `T` and `B * S`

`T` is the packed real-token count. `B * S` is padded capacity. Existing packed Attention paths use both:

- fused backend views can be governed by `T`;
- the unfused Q/K/V layout is governed by `B * S`; and
- PR1 retains the legacy `B * S` attention allocation total even when a fused inner view uses `T`.

Shrinking the total allocation from `B * S` to `T` is a separate optimization.

### Workspace components and layouts

PA has a projection GEMM allocation and an Attention allocation. The recipe reports them separately; their sum must
not replace either allocation. PMHA receives Q/K/V inputs and has zero projection workspace.

| Backend | Q/K/V representation | Q/K/V view dimension | Backend scratch retained by PR1 |
| --- | --- | --- | --- |
| Flash | Planar materialized views or direct input views | `T` | Softmax LSE: `sizeof(float) * B * S * N` |
| TensorRT fused (`FusedRunner`) | Interleaved `[T, N, 3, H]` | `T` | None |
| Memory-efficient Attention | Planar materialized views or direct input views | `T` | Optional FP32 accumulator: `sizeof(float) * B * S * N * H_v` |
| Unfused (`Default`) | Planar `[B, N, S, H]` views | `B * S` | Two individually aligned `element_size * B * N * S * S` regions |

The MEA accumulator is needed when `H_v > 128` and the input element size is smaller than FP32. PA does not dispatch
to Flash and always materializes Q/K/V after its projection GEMM. PMHA can skip materialization for packed
`[T, N, 3, H]` input on TensorRT when bias is absent, or for separate Q/K/V input on Flash or MEA when bias is absent.

### Validation and tests

PR1 includes:

- checked size, offset, alignment, and derived-stride arithmetic;
- CUDA int32 and int64 ABI validation scoped to the selected backend and materialization producer;
- explicit planar, interleaved, and direct-view contracts;
- recipe containment validation;
- independent hand-calculated byte and layout parity tests;
- runtime route tests for Flash, TensorRT fused, memory-efficient, and unfused paths as applicable; and
- empty-output handling before GEMM or CUDA kernel dispatch.

PR1 validates token-offset and cumulative-sequence tensor shapes and host-visible geometry. It does not inspect or
synchronize their device contents. Device-value validation requires a separate CUDA graph- and capture-safe contract.

### PR1 non-goals

PR1 does not:

- connect to the generic L1/L2 framework;
- change Attention backend selection or eligibility;
- change allocation count or lifetime;
- shrink the legacy `B * S` total to `T`;
- introduce planned-root allocation or preallocation; or
- validate device-side token-offset or cumulative-sequence values.

## Attention family rollout

PR1 precedes the sequence below and establishes the checked-recipe architecture that later families can reuse.

The planned operator-specific sequence is:

1. Shared backend primitives and `MultiHeadAttention`.
2. `GroupQueryAttention`.
3. `PagedAttention`.
4. High-value decoder-specific variants.
5. Separate memory models for Linear, Sparse, Longformer, quantized, and ONNX Attention operators.
6. Continue the generic planner integration after its budget and multi-slot contracts stabilize.

### GroupQueryAttention XQA and Flash recipes

The first GroupQueryAttention backend follow-up adds graph-free, checked
recipes for two concrete selected backends:

- XQA reproduces the semaphore, row-max, row-sum, and multi-block output
  accumulator layout from `GetXQAScratchSize`. Its backend allocation also
  retains the runtime's aligned RoPE Q/K buffers and dynamic FP32 head-sink
  conversion. A persistent prepacked head sink is not transient workspace.
- Flash retains the separate LSE, split-LSE, and split-output allocations.
  Fast decode selects splits using KV heads and the local-window-adjusted KV
  length, but sizes accumulators using query heads and a head size rounded to
  32, matching the GQA override.

Flash split selection is discontinuous and concrete workspace is not monotonic
in KV length. For example, with `B=1`, `S_q=1`, two heads, head size 64, and
108 SMs, increasing KV length from 13,824 to 13,825 changes the selected split
count from 54 to 28 and reduces the concrete split workspace. A future bounded
aggregate must therefore compute a conservative envelope over the bounded
domain or report unavailable; evaluating only the componentwise maximum is not
safe.

### GroupQueryAttention MEA, unfused, and complete-route recipes

The stacked route-composition follow-up adds the remaining selected backends:

- Memory-efficient Attention retains separate K and V head-expansion
  allocations at the effective staged cache capacity and its optional FP32
  output accumulator.
- Unfused retains the single combined allocation containing aligned Q BNSH,
  aligned Y BNSH, aligned FP32 QK, and the aligned softmax upper bound.

A complete concrete route places its preparation regions once, then places the
selected backend's simultaneously-live allocation regions at checked
256-byte-aligned offsets. This does not change runtime allocation topology and
does not select or aggregate routes. cuDNN is explicitly unavailable because a
graph-free recipe must not query or build a cuDNN graph.

MHA and GQA are high-value coverage targets and have high estimation-drift risk. Their runtime behavior can include
dynamic internal backend dispatch, cache lifecycle and aliasing, optional inputs, non-monotonic fallback paths, and
unfused workspace governed by `S_q * S_kv_total`. GQA additionally has different Q and KV head counts. MHA can have
different query and total-KV sequence lengths or different Q/K and V head sizes. Shared backend kernels do not
eliminate operator-specific preparation, cache, transpose, grouping, or output workspace.

Paged, Linear, Sparse, and Longformer Attention require distinct memory models. They must not be forced into a dense
Attention formula solely because they share some backend infrastructure.

## Acceptance standard

Each Attention family must provide:

- a runtime source of truth for the selected backend's workspace and layout;
- checked arithmetic and ABI, grid, alignment, and containment validation;
- byte-total and view-layout parity with runtime allocation;
- tests for reachable backend routes and fallback boundaries;
- explicit exact, safe-bound, or unavailable estimation semantics;
- no copied runtime dispatch cascade; and
- no dependency from the reusable recipe on graph or CUDA runtime types.

This document defines operator-estimation work tracks and invariants. It does not prescribe a public framework API.

## PA/PMHA framework follow-up

The PA/PMHA follow-up connects the graph-free recipes to the current Phase-A framework:

- **Level 1:** CUDA EP `GetCapability()` translates positional node inputs, uses max-shape inference when available
  (and graph metadata otherwise), enumerates routes that can be reached by some valid runtime geometry up to the
  supplied shape, and logs the aggregate. This estimate is
  currently **log-only**. It does not change the resource-accountant value or the partition accept/reject decision.
- **Level 2:** constructed PA and PMHA kernels translate `WorkspaceInputShape` entries and declare nonzero workspace
  slots. Missing mandatory inputs, present inputs without required shape metadata, partial required dimensions,
  malformed geometry, and checked overflow produce no declaration rather than a zero-byte estimate.
- **Zero-shaped hints:** `WorkspaceInputShape` does not identify whether a shape came from concrete graph metadata or
  max-shape inference. The adapter therefore treats any zero extent as unavailable and emits no declaration rather
  than interpreting it as a proven empty runtime output. The current Level-2 requirements boundary represents both
  unavailable estimates and explicit zero estimates as an empty requirement list, so it cannot expose that
  distinction. Exact-zero behavior remains available in the graph-free single-route recipe.

Route aggregation does not evaluate the runtime cascade only at the supplied maximum sequence length: dispatch gates
such as Flash and FP32 MEA thresholds or attention-bias alignment are non-monotonic across smaller runtime shapes.
Every backend that can be reached for some valid shape up to the supplied geometry is sized at that maximum geometry.
Unfused is always retained as a possible fallback for a nonempty problem, including alongside Flash, MEA, and
TensorRT candidates. Bound aggregates deliberately bypass exact equal-head recipe gates because an equal-head route
can be reachable below unequal maximum bounds; the recipe is still checked at those original maximum bounds. If a
newly reachable route's max-geometry recipe is invalid, or any included route otherwise cannot be safely bounded,
estimation is unavailable rather than silently omitting the route. Mutually exclusive route workspaces are combined
with `max`, never `sum`. The Level-1 aggregate describes one 256-byte-aligned, operator-owned root. PA places its
simultaneously-live projection and Attention regions in that root; PMHA naturally has only the Attention region:

```text
PA attention offset = align_up(projection bytes, 256)
PA root bytes = attention offset + max(reachable attention routes)
PMHA attention offset = 0
PMHA root bytes = max(reachable attention routes)
```

The PA alignment gap is intentional, so its root can be up to 255 bytes larger than the unaligned sum. Level 2 emits
exactly one slot-0 requirement for either operator, with explicit `alignment_bytes=256`. This does not depend on the
framework's generic multi-slot capability.

Runtime allocation topology is unchanged: PA continues to make separate dynamic projection and Attention
`GetScratchBuffer()` allocations, and PMHA continues to dynamically allocate its Attention workspace. Future #32071
integration must atomically add the explicit `SupportsPreallocatedWorkspace()` planner opt-in, retrieve the planned
root, and slice it at the declared Attention offset. A declaration alone is not planner opt-in and does not itself
change runtime allocation behavior.
