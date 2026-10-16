# Policy-Based, Route-Aware Workspace Estimation and Declaration

## Context

This document extends [`future_directions_constrained_env.md`](future_directions_constrained_env.md)
(Direction 2, "Minimize Allocations") and the
[`cuda_kernel_workspace_inventory.md`](cuda_kernel_workspace_inventory.md). It addresses a
specific failure mode in Level-1 (L1) partition-time workspace estimation that blocks smart
CPU offloading on VRAM-constrained machines.

The motivating goal is unchanged: let ORT decide **which nodes/layers run on GPU** under a VRAM
budget, spilling the rest to CPU, without over-reserving VRAM so conservatively that heavy nodes
are needlessly pushed off the GPU.

### The dilemma this design resolves

We want two things from a partition-time estimate that pull in opposite directions:

1. **Safety** — the per-node estimate must upper-bound that node's admitted workspace.
2. **Tightness** — the estimate must be close to actual, so we do not waste VRAM and evict
   heavy nodes to CPU that would have fit.

Forcing a *single static number* to be simultaneously a safe upper bound and a tight value is
impossible for a dynamic-shape op whose kernel can take several routes with very different
footprints. "Conservative" wins, VRAM is wasted, and utilization drops.

### Proven bounds, empirical peaks, and heuristics

These are different kinds of evidence and must not be interchanged:

- A **proven upper bound** follows from formulas that cover every admitted state and every reachable
  route. It is suitable for a capacity decision only while its route and workload assumptions are
  enforced.
- An **empirical peak** is the largest allocation observed during one or more profiling Runs. It is
  useful for validating a bound and calibrating cost models, but an unobserved shape, route, allocator
  state, or concurrent lifetime can exceed it. A sampled peak is therefore not a proof bound.
- A **heuristic** is an assumed margin without either proof or exhaustive measurement. The generic
  resource-accountant fallback is in this category: it estimates workspace as 50% of the known
  initializer-plus-output bytes. It is neither a GQA route bound nor evidence that placement cannot
  OOM.

### The concrete symptom (GQA)

`GroupQueryAttention` (CUDA) can dispatch to several kernels — XQA, cuDNN SDPA, FlashAttention,
Memory-Efficient Attention (MEA), or an unfused fallback. Their workspace footprints differ by
orders of magnitude:

| Route | Workspace scaling | Reference |
|---|---|---|
| Flash (prefill) | **linear** in `sequence_length · num_splits` | `group_query_attention.cc` flash split buffers |
| Flash fast-decode | **~zero** scratch | `group_query_attention_impl.h` (`GQABufferRequirements::Compute` returns all-zeros) |
| XQA (decode) | **linear** in KV capacity | `group_query_attention.cc` `GetXQAScratchSize(...)` |
| MEA | linear in KV capacity (head expansion) | `group_query_attention.cc` MEA `kv_buffer_bytes` |
| **Unfused fallback** | **quadratic**: `2 · sizeof(float) · B · N_q · S_q · S_kv` | `unfused_attention.cu` `GetUnfusedAttentionWorkspaceSize` |

A route-unaware L1 estimate that bounds the **unfused** route reserves quadratic scratch even
though the runtime almost always selects **flash** (linear). Measured effect: the static
preallocation path (`session.enable_static_workspace_preallocation=1`) reserved ~123 MB where
flash actually consumed ~3.2 MB at 1024 context (~38x over-reservation), and the reservation
grew ~4x per context doubling (the quadratic signature). On a 24 GB card this pegged memory at
8K/12K context — a *regression* versus the default scratch path.

**Root cause:** the estimate reflected a route the runtime never runs. The fix is not "a better
number" — it is to make estimation *route-aware* and *policy-bound*.

---

## Core idea: estimate against a committed dispatch policy

### Definitions

- **Route (kernel):** one implementation (Flash, XQA, cuDNN SDPA, MEA, unfused, or a future
  bounded fallback).
- **Runtime state `x`:** the per-Run inputs that steer dispatch —
  `(phase, S_q, S_kv, dtype, device, flags)` where `S_q = sequence_length` (query axis) and
  `S_kv = total_sequence_length` / `seqlen_present_kv_cache` (KV axis).
- **Dispatch intent `I`:** the portable user objective (`auto`, `latency`, `memory`, or `safe`).
  It is not itself a route and has no hardware-specific meaning until an EP resolves it.
- **Committed policy `P`:** a **total, enforceable** function `P: x → non-empty set<route>`.
  It restricts the runtime to a known route set for every admitted state. A singleton set pins a
  route; a larger set preserves bounded fallback freedom.
- **Runtime selector `π`:** the existing or future ranking logic constrained so
  `π(x) ∈ P(x)`. The selector may choose differently per phase and shape, but it may never escape
  the committed set.

### The L1 requirement under a policy

```
W_L1(node) = max over admitted states x and routes r ∈ P(x): CompleteWorkspace(x, r)
```

`CompleteWorkspace` is the route's **full** footprint (e.g. `qkv_buffer` + the route's auxiliary
buffers), not one term. The maximum is taken over the executions the **committed policy** admits —
not over every implementation in the source tree.

### Why a policy, not a single kernel or a scalar `speed_rank`

A single node serves **both** prefill (`S_q = N`) and decode (`S_q = 1`), and those phases
legitimately dispatch to *different* kernels. So:

- You cannot bind one route to a node. The binding unit is **`(node, phase)`**.
- The "fastest route" is **shape/phase-dependent** (decode is memory-bound → XQA wins; prefill is
  compute-bound → Flash wins). A single static `speed_rank[route]` collapses a function into a
  constant and mis-serves one phase. The policy `π(x)` keeps preference a *function of state* —
  its branches are the pieces of a piecewise preference.

This is exactly TensorRT's **optimization-profile** model: a different tactic is tuned per shape
regime, and the workspace reservation is the max across profiles.

### Route descriptors are functions, not constants

Each runtime route needs a descriptor whose properties are evaluated for the workload and device:

```
RouteDescriptor {
  eligible(x, device) -> bool
  workspace_bound(envelope, device) -> bytes | unavailable
  latency_rank(x, device) -> comparable cost or ordered preference
  proof_kind -> analytical | empirical | heuristic
}
```

Workspace and latency are not scalar route attributes. Both depend on shape, phase, dtype, device,
and feature flags. The first GQA policy implementation therefore uses analytical workspace
functions but deliberately does **not** claim a calibrated latency model: `latency` reuses the
current runtime priority order. Empirical timing can replace that ordering after it is
collected over the declared workload envelope; timing samples do not establish the envelope bounds.

### The three properties that make `P` the estimation object

1. **Total over the admitted workload.** `P(x)` must be non-empty for every admitted `x`. If some
   admitted state maps nowhere (and the fallback is prohibited), the policy is **invalid** — narrow
   the admitted set or add a bounded fallback. This is the coverage condition.
2. **Bounded image.** Every route in every `P(x)` has a complete analytical bound. A policy may
   include unfused when its quadratic bound is acceptable; it may exclude unfused only when the
   remaining permitted set is total over the admitted workload.
3. **Enforceable — estimation and execution are one commitment.** The runtime must dispatch to
   some route in `P(x)`. A would-be out-of-policy route is a *policy violation* → explicit error,
   **never** a silent allocation outside the bound.

> **Soundness invariant:** an L1 estimate is sound only if it bounds *every route the runtime can
> actually execute*. A route may be dropped from the `max` **iff** the runtime is incapable of
> running it. Excluding unfused from the estimate while the runtime can still fall to unfused is
> *unsound* and will OOM on exactly the Run where unfused fires.

### The contract: bind constraints, not a kernel

The unit of commitment is not a chosen kernel but a **contract** — the triple:

```
contract(node) = ( admitted workload = envelope,
                    permitted routes  = image(P),
                    workspace bound    = W_L1 )
```

Inside that contract the runtime keeps **full freedom**: it may pick any permitted route for any
admitted state, and pick differently per phase and per shape. The rigor this forces is that **`W_L1`
must be *closed under* that residual freedom** — it is the `max` over *every* route the contract
still permits for each admitted state, never the single route we would predict. If the contract
permits `{flash, XQA}` for decode, L1 budgets `max(flash, XQA)` for decode, then `max` across phases
(single-phase Run) or `sum` (if a Run ever mixes phases). "Bind constraints, not kernels" therefore
translates exactly into "**L1 maxes over the permitted set**," and this is what dissolves the
prefill-vs-decode different-route problem: the contract permits different routes per phase *by
construction*.

---

## Route-awareness: proving the expensive route excluded

The unfused fallback fires only when **all** fast routes are ineligible
(`group_query_attention.cc`, unfused guard):

```cpp
if (!use_xqa && !use_cudnn_sdpa && !use_flash_attention && !use_memory_efficient_attention &&
    !is_inputs_quantized && !use_smooth_softmax && head_sink == nullptr && past_kv_format == Q_K_V_BNSH)
  data.use_unfused = true;
```

The estimator does **not** need to predict which route runs. It needs to prove **at least one**
fast route is on for every admitted Run. Two cases:

### Case A — Flash or MEA is *statically* eligible → unfused already unreachable

Both eligibility predicates are **sequence-length-free** — pure functions of partition-time
constants:

```cpp
flash::is_supported<T>(device_prop, head_size, num_heads, kv_num_heads)      // no seq args
has_memory_efficient_attention(sm, fp16, bf16, head_size, head_size)         // no seq args
```

By the priority ladder, on any Run either a higher-priority route claims it (and runs, linear), or
— if none does — Flash/MEA runs *because it is statically eligible*. Either way one of
`use_flash || use_mea || use_xqa || use_cudnn` is true, so the unfused guard is false. **Unfused
never executes in the current source, with no code change.** This is the common production case
(fp16/bf16, head_size 64/128, SM80+, no bias).

Case A eliminates only the *unfused* terminal; it does **not** by itself bound the higher-priority
routes. In particular a reachable **cuDNN** route (preferred on SM≥90, including auto-selection — the
cuDNN-SDPA block in `group_query_attention.cc`) has no graph-free workspace oracle, so
`max(prefill, decode, chunk)` — which sizes only Flash/XQA — is sound **only when cuDNN is not
reachable**. The estimator enforces exactly this: `BuildBounds` flags a reachable cuDNN route
(`group_query_attention_workspace_estimate.cc`) and `GetGQAWorkspaceAggregateForBounds` returns
`Unavailable` for it (`group_query_attention_workspace_bounds.cc`), so the reduced estimate is
emitted only when **every** reachable route is soundly sizeable. Under capacity-aware CUDA
partitioning, an unavailable GQA estimate must fail closed by declining CUDA placement for that node.
The framework's generic 50% fallback remains available to unrelated operators, but it cannot
substitute for a missing GQA proof bound.

Inputs to the predicate and where they come from at partition time:

| Input | Source | Per-Run? |
|---|---|---|
| `T` (fp16/bf16/fp32) | node tensor dtype | no |
| `device_prop` | EP owns its device before `GetCapability` | no |
| `head_size`, `num_heads`, `kv_num_heads` | static Q / `past_key` shape dims | no |
| `has_attention_bias` | optional input-edge presence | no (graph structure) |
| `disable_flash_attention_` etc. | build flag + session `AttentionKernelOptions` | no |

### Case B — Flash *and* MEA both statically ineligible → unfused genuinely reachable

The only fast routes left are **XQA** (decode-only: `!is_first_prompt && sequence_length == 1`) and
**cuDNN** (phase-*dependent*: its `is_supported` takes `sequence_length` / `seqlen_present_kv_cache`).
In the **prefill** phase XQA is ineligible; unless cuDNN covers prefill, **unfused fires**.

Here a sound estimate **must** return the unfused quadratic worst case — *unless* the runtime is
changed so unfused cannot execute:

- **Prohibit unfused** by replacing the fallback with (a) a **hard error** ("no admitted route for
  this state"), or (b) a **bounded fallback route** (tiled attention, workspace ≤ fixed budget).
- Only after that change is unfused out of `image(P)`, and only then may the `max` drop it.

**A bounded fallback fills any coverage gap left by the remaining fast routes.** It is needed for
successful service of those states, not for a fail-closed contract that explicitly rejects them.
It is the ORT analog of cuDNN's zero-workspace algorithm. ORT's current fallback is the
*expensive* route, so additional low-workspace coverage is useful but can be deferred while the
framework plans against existing enforceable contracts.

---

## The envelope: two independent axes

Precision for a dynamic dim requires an externally declared operating envelope. Two axes, supplied
separately:

| Axis | Meaning | Source at estimate time |
|---|---|---|
| **`S_q` (query)** | tokens fed this Run (`N` prefill, `1` decode, chunk size) | `session.max_shape_override` on `input_ids` |
| **`S_kv` / capacity** | accumulated KV context / provisioned ceiling (`seqlen_present_kv_cache`) | overridden `past_key` seq dim, or the GQA capacity knob (see below) |

`S_kv` (`total_sequence_length`) is a **runtime scalar input** — not recoverable from graph shapes —
which is why a *proposed* out-of-band knob (`ep.cuda.gqa_workspace_max_total_sequence_length`, added
by the envelope-plumbing PR in this series) hands the estimator the KV bound directly. The ordering
`S_q ≤ total_sequence_length ≤ kv_cache_capacity` holds, and the two axes coincide only in the first
full prefill.

**Two caveats the envelope must respect:**

- **`max_shape_override` is an estimation *hint*, not a runtime bound.** Its public contract
  (`onnxruntime_session_options_config_keys.h`, `kOrtSessionOptionsMaxShapeOverride`) states it does
  not constrain runtime inputs, so a valid Run whose `S_q` exceeds the declared `P` can still exceed
  `W_L1`. The no-OOM claim therefore requires a **hard runtime envelope check** on *both* the query
  and KV bounds — a separate enforced option, not `max_shape_override` alone — or the smaller
  estimate must not be trusted.
- **The KV axis is cache-mode-aware.** For sliding-window GQA the *absolute* `total_sequence_length`
  may exceed the resident cache capacity, and workspace is sized from a route-specific
  resident/staged extent (`GetGQAEffectiveWorkspaceKvLength`), not from absolute context length. The
  envelope's KV bound must be that effective extent for a windowed cache and the absolute total
  length for a non-windowed cache; conflating the two would reject valid windowed workloads or derive
  the wrong bound.

See [`onnxruntime_session_options_config_keys.h`](../../include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
`kOrtSessionOptionsMaxShapeOverride` ("session.max_shape_override").

---

## Phase handling: L1 never detects the phase — it enumerates

Phase is a per-Run property (`sequence_length` vs `total_sequence_length`, see
`group_query_attention_helper.h`: `is_first_prompt = (sequence_length == total_sequence_length)`,
`is_subsequent_prompt = (sequence_length > 1 && sequence_length != total_sequence_length)`). At
partition time no Run has happened, so L1 has no phase to observe.

**Resolution: phase is a quantified variable in the `max`, not an observed input.** L1 enumerates
the phases the envelope admits and evaluates the policy at each phase-corner:

```
W_L1 = max over p ∈ AdmittedPhases and r ∈ P(p):
         CompleteWorkspace(corner(p, envelope), r)
```

At runtime the GQA kernel uses the same partition: `is_first_prompt` selects prefill, a non-first
single-token step selects decode, and any other multi-token step selects chunked prefill. When
`total_sequence_length` resides on the GPU, `CheckInputs` cannot read it during graph capture and
leaves both prompt flags false. A single-token first prompt therefore uses the **decode** policy mask,
and a multi-token first prompt uses the **chunked-prefill** mask. These are runtime dispatch classes,
not proof of the semantic prompt stage. L1 must admit and size every class this classifier can select:
prefill and decode for a positive query-length bound, plus chunked prefill when the bound exceeds one.
The current #33316 adapter already admits those classes without requiring a device-to-host scalar read.

Everything L1 needs is static:

| Ingredient | Source |
|---|---|
| which phases are admissible | runtime-classifier reachability under the query-length bound, including GPU-resident total-length behavior; graph edge presence alone does not prove prefill-only execution |
| query bound `P` | `max_shape_override` on `input_ids` |
| KV capacity `C` | overridden `past_key` seq dim / GQA capacity knob |
| routes per phase `P(p)` | the committed policy, evaluated symbolically |
| route eligibility | seq-free static predicates (dtype, device_prop, head_size, heads, flags) |

The current `BuildBounds` requires both `past_key` and `past_value` shapes for cache geometry/capacity
and declines a no-past input configuration. It does not implement a no-past prefill-only estimate.
Future no-past support must establish its own allocation bound and cover the actual runtime classifier,
or enforce an explicit phase-admission condition before execution; it must not simply omit decode/chunk
from the mask based on absent graph inputs.

### `max`, not `sum` — and the one caveat

One GQA invocation selects one policy phase and one backend route. Those alternatives are mutually
exclusive, so its workspace bound is their **max**, not their sum. Sequential Runs can reuse backing
under the declared serialization/allocator assumptions. This does not establish that every GQA node
in an entire Run has the same phase.

For CPU-resident `total_sequence_length`, GQA checks
`batch_size must be 1 when sequence_length > 1 and past context is given`
in `group_query_attention_helper.h`. The GPU-resident branch skips that prompt-type block, including
the batch-size guard. Do not use this guard as a universal proof of single-phase batching or
cross-node phase correlation; size the actual batch bound for the reachable dispatch class.

> **Caveat:** if ORT ever adopts vLLM-style **continuous batching / mixed chunked-prefill + decode
> in one forward** (ragged batch), prefill and decode routes become simultaneously live in one Run,
> the plan must account for overlapping `(phase → route)` workspaces according to their real
> allocation identities/lifetimes. A maximum over mutually exclusive routes does not cover such
> simultaneous execution.

---

## Declaration and enforcement (Level-2)

The L1 budget must be reconciled with what the kernel actually declares/consumes:

- **L1 (partition):** `W_L1 = max over admitted (phase × route) of CompleteWorkspace`, used by
  `IResourceAccountant` for placement. `WorkspaceEstimateSource`
  (`include/onnxruntime/core/framework/resource_accountant.h`) records provenance, not proof:
  `kFallback` is the generic 50% heuristic, `kEstimator` may be a proven analytical route bound, and
  `kProfile` is an empirical observation. `kProfileAndEstimator` retains the maximum of the observed
  and analytical values. Capacity-aware GQA placement accepts only an available route-aware
  analytical bound; neither `kFallback` nor `kProfile` repairs an unavailable proof.
- **L2 (Initialize):** `DeclareWorkspaceRequirements` declares the concrete per-slot need for the
  chosen envelope. `session.strict_workspace_verification` governs whether an L2 declaration larger
  than the L1 reservation fails Initialize (`=1`, strict) or logs and retains dynamic allocation
  (`=0`, default).
- **Single-source the sizing.** The recipe system in
  `group_query_attention_workspace*.{h,cc}` (`GetGQAPreparationRecipe` plus per-backend
  `GetGQA{Xqa,Flash,MemoryEfficient,Unfused}WorkspaceRecipe`, composed by
  `GetGQACompleteWorkspaceRecipe`, overflow-checked and validated) drives both the partition-time
  estimator and runtime allocation. The GQA-first committed-policy slice also passes the validated
  KV envelope and dispatch intent into `DeclareWorkspaceRequirements`, resolves the same committed
  prefill, decode, and chunked-prefill route masks, and stores those masks in the kernel for
  phase-aware runtime enforcement. Route ranking is
  canonicalized across L1 and L2 when their knowledge differs (for example, L1 knows a constant
  head sink *may* be prepacked while L2 knows the exact post-PrePack state). Sharing formulas makes
  each concrete Run's allocation provably **bounded by** `W_L1`; it does not make them equal because
  `W_L1` is the maximum over envelope states and permitted routes, while a Run takes one concrete
  state and route.

### Requirements vs. allocation strategy — and the lifetime invariant

L1 answers **how much** (the node's peak-concurrent requirement across its policy); L2 answers **how
it is provided** (a persistently preallocated slot vs. transient scratch/arena). These must stay
separable so allocation strategy can be tuned by benchmark without moving the placement number — the
benchmark lesson (do *not* persist prefill-dominated GQA; *do* persist phase-invariant MatMulNBits) is
an L2 decision that should not perturb L1.

The invariant that keeps them separable:

> **L2 may vary a buffer's *source* freely without touching L1, but changing its *lifetime* changes
> L1's concurrency.** A persistently preallocated decode buffer is live *during* prefill, so it
> stacks with prefill scratch — L1 must then budget `prefill_scratch + persistent_decode_slot`, not
> `max(prefill_scratch, decode_slot)`. If L2 changes lifetimes, L1 re-budgets under the declared
> lifetime model.

`strict_workspace_verification` remains the guardrail that an L2 declaration never exceeds the L1
reservation.

### Per-node bounds are not the whole-graph peak

`W_L1(node)` is a local contract. A whole-session VRAM bound additionally needs the execution plan's
lifetimes:

```
GlobalPeak = persistent residents
             + max over execution time (live activations + concurrently live workspaces)
             + allocator/runtime reserve
```

This is not generally `sum(W_L1(node))`. Sequential scratch can reuse storage, while persistent L2
slots overlap every Run; parallel streams, control-flow subgraphs, and allocator retention can create
other overlaps. The current accountant's additive node charges do not perform this liveness analysis.
Whole-graph peak/lifetime accounting is separate follow-up work. Route-aware per-node bounds are
required inputs to that planner, not a replacement for it; additive accounting is conservative only
when every local charge is itself a proven bound and all non-node reserve is represented.

---

## User configuration

Choosing a policy is choosing a **contract**, so the surface is "curated menu + validation", not
free authorship. Three actors:

| Actor | Role |
|---|---|
| **Kernel author** | defines the *menu* of named policies, each with an admission predicate + route→workspace map |
| **User / deployment** | *selects* from the menu (or supplies budget + intent); optional expert override |
| **Framework** | *validates* the selection against model + device + envelope; auto-selects by default |

### Three tiers

**Tier 0 — auto (default).** The GQA-first implementation preserves the current runtime route
priority and behavior. In capacity-aware placement, L1 still declines when any reachable route
(currently cuDNN) lacks a graph-free workspace oracle. Future auto-selection can compare validated
contracts against budget and measured cost; it must not silently turn an unproven route into a
capacity proof.

**Tier 1 — intent-level named policy (portable session option).**

```
session.kernel_dispatch_policy = "latency" | "memory" | "safe"
```

Intent, not mechanism — portable across ops/EPs. The first GQA implementation resolves it as:

- **`latency`:** preserve the current route priority
  (XQA → cuDNN → Flash/fast-decode → MEA → unfused). This is an ordered heuristic, not a measured
  latency model. Decode-only routes are absent from the prefill and chunked-prefill masks. If a
  permitted route has no analytical workspace oracle, strict capacity-aware L1 remains unavailable.
- **`memory`:** among analytically bounded routes, choose the smallest bound independently for
  prefill, chunked prefill, and single-token decode. The selected general-phase routes are also
  admitted as decode fallbacks for runtime-only failures such as an XQA shared-memory rejection,
  but decode-only routes do not leak into either general phase. L1 takes the maximum over every
  route in each phase mask, then the maximum across phases.
- **`safe`:** for each phase, permit every applicable reachable route with a complete analytical
  bound and exclude unproven routes such as cuDNN. This maximizes bounded fallback coverage; it
  does not mean "always choose the slowest route."

For every non-auto committed policy, if no permitted route serves a Run, runtime returns an explicit
error rather than escaping to an unaccounted route.

```
latency : minimize per-kernel time       → larger reservation → fewer resident nodes → more offload
memory  : minimize per-node workspace     → more resident nodes → less offload
safe    : maximize proven route coverage  → predictable route-local bound
```

**`auto` remains the zero-config default and preserves existing dispatch.** Explicit `latency`
uses that priority and suits a single interactive session whose model comfortably fits in VRAM.
The reason the other two intents exist is
that this knob is **not only** about per-kernel speed: it changes how much workspace each node
reserves, which changes how many nodes the capacity accountant keeps resident vs. offloads to CPU.
CPU offload is **asymmetrically expensive** (PCIe round-trip + slow CPU matmul, often 10–100× the
GPU kernel), and that asymmetry is what flips the choice:

- **Choose `memory` near the VRAM cliff.** If `latency`'s high-workspace routes push even one extra
  layer to CPU, that layer's offload cost can exceed the aggregate kernel-speed savings across all
  resident layers, so `memory` is **faster end-to-end** despite each kernel being individually
  slower. This is the central case the offloading design targets. Also the right pick for
  **multi-tenant / high-concurrency** GPUs (minimize per-session footprint so more sessions stay
  resident) and **large-batch / long-context serving** (spend VRAM on batch/KV rather than scratch).
- **Choose `safe` when a route-level proof is required.** It removes routes without a graph-free
  oracle (currently cuDNN) while retaining every proven fallback, including unfused when eligible.
  A system-wide no-OOM guarantee additionally requires guaranteed query/KV bounds, whole-graph
  peak/lifetime accounting, and explicit allocator/runtime reserve; those are not supplied by a
  route policy alone. Bounds may come from ORT validation or a caller's admission contract.
  `safe` is also useful for reproducibility, debugging,
  and portability across devices with different fused-route availability.

Users switch away from `latency` precisely when the **system-level** objective (fit, concurrency,
or a proof-oriented deployment) outweighs the **node-level** objective (fastest kernel):
interactive-and-fits → `latency`; constrained-or-shared → `memory`; bounded-route / debugging →
`safe`. These three named intents are the **manual approximation** of the Tier-0 auto-optimizer that
consumes the `(W_L1, cost)` pair per route and solves the offload frontier automatically; shipping
the intents first lets most users pick the right bucket by hand without the full cost model.

**Tier 2 — explicit route menu (EP-specific provider option, expert).**

```
ep.cuda.gqa_dispatch_policy = "prefill=flash;decode=xqa;fallback=error"
```

The literal `π`, for benchmarking / power users. EP-scoped because routes are hardware-specific —
consistent with the proposed `ep.cuda.gqa_workspace_max_total_sequence_length` (added by the
envelope-plumbing PR in this series). Today's scattered
per-route toggles become the **compile target** this policy lowers to:

- `ORT_ENABLE_XQA`, `ORT_DISABLE_FLASH_DECODE` — environment variables
  (`group_query_attention.cc`)
- `AttentionKernelOptions` via `GetAttentionKernelOptions()` — session-scoped
  (`UseFlashAttention()`, `UseEfficientAttention()`, `UseCudnnFlashAttention()`)
- model attributes (`num_heads`, `local_window_size`, `softcap`, …)

### Validation is non-negotiable (regardless of who chose)

At Initialize, run a **coverage check**: is `P` total over the admitted `(phase × envelope)` on
*this* device? Reuse the strictness pattern already modeled by
`session.strict_workspace_verification`:

- The GQA-first implementation fails Initialize for `memory` or `safe` when the declared shapes and
  envelope cannot resolve a complete policy.
- At Run time, a per-Run availability failure may fall back only to another route in the committed
  set. If none is available, the Run fails explicitly.
- A future permissive mode may degrade only to a fallback already represented in `P` and `W_L1`;
  it must never enlarge the route set after accounting.

Explicit choice ≠ unchecked choice.

### Orthogonal knobs and scope

- **Budget is separate from policy** (cuDNN workspace-limit / vLLM `gpu-memory-utilization`
  precedent): the user supplies the VRAM budget/reserve; the policy defines the route *function*;
  ORT checks `W_L1(P) ≤ budget`.
- **Session-scoped by default.** All attention nodes share one KV-cache envelope, so `P` is one
  policy for all GQA nodes (matching `AttentionKernelOptions`), with an optional per-node
  **attribute** override.
- **Safe for output correctness.** The routes are numerically equivalent to fp tolerance, so
  choosing a policy changes *memory/latency, not model outputs*. The policy is a Pareto-frontier
  lever tying directly to the offloading tradeoff: `memory` → smaller `W_L1` → more nodes on GPU.

### Do not statically preallocate prefill-dominated ops

GQA is prefill-dominated: its workspace scales with query `sequence_length`, so a persistently
preallocated buffer sized for prefill is held through the decode-heavy steady state.
`session.enable_static_workspace_preallocation` should target **phase-invariant** ops (e.g.
MatMulNBits), leaving GQA on the transient arena/scratch path. This is precisely why the benchmark's
`combined` mode regressed.

---

## How other frameworks resolve the multi-route problem

| Framework | Multi-route resolution | Per-route cost known? | Cheap fallback? | Binding |
|---|---|---|---|---|
| **cuDNN** | budget → pick affordable route; re-query per call | yes (`getAlgorithm_v7` list) | **yes** (zero-ws algo) | per call |
| **TensorRT** | workspace-pool limit prunes tactics at build; optimization profiles per shape regime | yes (`getWorkspaceSize`) | yes (fallback tactic) | **frozen in engine** |
| **llama.cpp / ggml** | one kernel per op; measure whole graph at worst-case (`ggml_gallocr_reserve`, liveness) | n/a | n/a | reserved once |
| **vLLM** | pin backend (`VLLM_ATTENTION_BACKEND`), then `profile_run` measures peak; KV = budget·util − weights − peak | no (measured) | n/a | fixed at init |

Two archetypes:

1. **Constrain-then-select** (cuDNN, TensorRT): fix a budget, **exclude** routes that do not fit,
   take the fastest survivor. Requires a queryable per-route size **and a guaranteed low-workspace
   fallback**.
2. **Pin-then-measure** (ggml, vLLM): commit the route (or whole graph), measure its real footprint
   once at the envelope.

ORT's generic resource-accountant fallback does **neither**: it assigns workspace equal to 50% of
known initializer-plus-output bytes, without modeling routes or proving a bound. The route-aware GQA
estimator instead uses a **static route map + analytical envelope bound**. An envelope profiling Run
then validates that bound and calibrates performance/cost models; it never replaces the analytical
bound for a strict capacity decision. Heed vLLM's documented pitfall: even a nominal maximum-shape
profile can miss a larger peak from another admitted regime (such as chunked prefill), and a dummy
Run can itself OOM before yielding a useful peak.

---

## Offload candidate selection

Workspace estimation (this document) is **one of two independent VRAM levers**; a complete
capacity-aware offloading design uses both:

- **Persistent weight residency** — where a node's *parameters* live (GPU VRAM vs. host RAM). This
  is the dominant consumer for most real-world OOMs and the highest-ROI place to start.
- **Transient workspace / activation** — the per-Run scratch a resident node needs. This is what the
  route-aware `W_L1` estimate bounds, and it is what lets you *also* fit more **compute-bound** nodes
  once the weight-side wins are taken.

They are separate accounting: moving a weight to CPU reduces persistent footprint; a tighter `W_L1`
reduces the transient peak reserved alongside the resident set.

### What makes a node a good offload candidate

A node is a good candidate to push to CPU when it hits all four properties — in which case the VRAM
freed is large and the CPU penalty is small:

1. **Large weight footprint** → big VRAM savings per offload.
2. **Low arithmetic intensity (memory-bound)** → the GPU's FLOP advantage is wasted anyway, so the
   CPU compute penalty is small.
3. **Low execution frequency** (not every layer, or sparsely activated) → the offload cost amortizes.
4. **Small activation transfer across PCIe** relative to weight size → cheap I/O.

| Node | Weight size | Compute / byte | Frequency | Offload? |
|---|---|---|---|---|
| Embedding / `lm_head` | very large (`vocab × hidden`) | ~0 (gather) | once / token | ✅ best |
| MoE / QMoE experts | largest block | low (sparse top-`k`) | sparse | ✅ best |
| Attention (GQA) | small | high | every layer | ❌ keep resident |
| Dense FFN | medium | high | every layer | ❌ keep resident |

**Embeddings** are a gather (near-zero FLOPs) over a multi-GB `vocab × hidden` table that runs once
at each I/O boundary; the gathered rows shipped over PCIe are tiny. **MoE/QMoE experts** are the
largest parameter block (often 80–95% of total params) but each token touches only top-`k`, so stored
weight dwarfs per-step FLOPs — exactly the "GPU residency wasted" profile. **Attention and dense FFN**
hit *none* of the four properties (compute-bound, every layer, high activation traffic) and stay
resident — attention is the critical path these workspace estimates are about.

### The practical knob

The first-cut control is a **node count per VRAM tier**, not a byte budget — e.g. "offload the first
`N` QMoE layers; `N=8` for an 8 GB GPU, `N=0` for 16 GB+", set from a runtime/GenAI profile. This is
coarse but effective (a handful of named nodes hold most of the memory), trivial to profile per tier,
and it **sidesteps the per-op workspace-oracle problem** entirely: keep the compute-bound attention
resident and offload the cheap-to-move weight-heavy nodes. In ORT the target is the fused
`com.microsoft.MoE` / `QMoE` node's packed expert-weight **initializer inputs**
(`fc1/fc2/fc3_experts_weights`) — one node per layer — so "offload `N` MoE layers" is literally
redirecting the weight inputs of `N` `QMoE` nodes to a CPU allocator. The structure mirrors the
[three-tier config](#three-tiers): a portable count/intent (`session.offload_moe_layers = N`)
lowering to an EP-specific tensor-placement rule, profiled per device.

This is well-precedented on the weight lever: llama.cpp's `--n-cpu-moe N` / `--override-tensor`
(regex `exps=CPU`), the `mixtral-offloading` project, DeepSpeed ZeRO-Inference weight offload, and HF
`accelerate` `device_map="auto"` all offload exactly these tensors. Ship the weight-placement option
as the high-ROI starting point; the workspace estimate composes on top to reclaim headroom among the
nodes that *stay* resident.

---

## Worked example: strict GQA policy `P_GQA`

```
Policy P_GQA(x):
  prefill  (is_first_prompt)          → {Flash}       if flash_supported else INVALID
  decode   (seq==1 && !is_first)      → {XQA, Flash}  after eligibility filtering
  chunk    (1 < seq < total)          → {Flash}       if flash_supported else INVALID
  # unfused ∉ image(P_GQA)  →  prohibited (error or bounded fallback)
```

Envelope: caller-declared query bound `P` (estimated from `max_shape_override`) and KV capacity `C`
(from the GQA knob). This is a conditional estimation contract unless the bounds are guaranteed by
the caller or ORT admission/validation. Even guaranteed workload bounds require global accounting
and allocator/runtime coverage before they imply a session no-OOM claim.

```
prefill/chunk :  CompleteWorkspace(S_q=P, S_kv=C, Flash) = lse + lse_accum + out_accum
decode        :  max(CompleteWorkspace(XQA), CompleteWorkspace(Flash))                    [fallback-closed]
W_L1 = max(...) = flash prefill term        ← ~38x below the unfused reservation
```

The Flash term is **not** read off a single envelope corner. Its `lse_accum`/`out_accum` size scales
with `num_splits`, which `flash::num_splits_heuristic` picks by **occupancy**: a large `S_q=P` may use
one split (no accumulators) while a *shorter* admitted chunk at the same KV extent uses several, so
the maximum need not sit at `(S_q=P, S_kv=C)`. The estimator bounds it across **all** occupancy
regimes — `FlashBackendEnvelope` (`group_query_attention_workspace_bounds.cc`) sizes the accumulators
with the *structural* split ceiling
`max_splits = min(128, multi_processor_count, ⌈(kv + block_n − 1) / block_n⌉)` rather than evaluating
the heuristic at a point ("sound across its discontinuities"); preparation/QKV buffers come from
`GQABufferRequirements`. Using `(S_q=P, S_kv=C)` for the *shape* axes still over-approximates any
individual Run (first prefill has `S_kv=S_q`; a deep chunk has small `S_q`). If a policy branch
switches route at an interior shape threshold, take the `max` **piecewise per route-region** (the max
may sit at the switch boundary, not the envelope corner). Later profiling *validates* this bound; it
does not replace it with a sampled peak.

---

## Comparability: one contract → a Pareto set of contracts

The design's natural end-state elevates each node from *one* contract to a small **Pareto set** of
contracts, each exposing a `(W_L1, latency)` pair. After the separate whole-graph lifetime work,
placement can solve a **global** objective — minimize model latency subject to
`GlobalPeak(placement, policy) ≤ capacity` — instead of greedily fitting each node independently.

The effect that makes this non-local: a **compact-but-slower** GPU policy can beat a
**fast-but-larger** one, because the fast policy may evict a neighbor to CPU, and the **CPU boundary
(H2D/D2H transfer + slow host compute) is an edge cost, not a node cost**. Modeling boundary
crossings couples neighboring placement decisions — it is a knapsack/assignment problem, not a set of
independent choices.

**Honest dependency:** a claim of minimum end-to-end latency needs comparable measured or calibrated
costs, including CPU-boundary costs. However, framework budget selection need not wait for that
calibration: an initial selector can choose feasible contracts using deterministic ordinal
preferences and report that its performance objective is heuristic. Keep cost evidence explicitly
tagged as measured, modeled, ordinal, or unavailable; never turn a route's ordinal rank into
microseconds or add unrelated ranks as if they were calibrated times.

### Automatic selection is this comparison, run for the user

The Tier-0 end state is this global comparison solved on the user's behalf: the user
declares **budget + workload envelope + intent (`latency`/`memory`)**, and the framework enumerates
the **kernel author's** policy menu, evaluates each policy's `(W_L1, cost)` for their device and
envelope, and picks the placement+policy that meets budget with the best objective. The user never
authors a dispatch table; the kernel author owns the menu, the framework owns the selection, and
explicit `fused_only`-style overrides remain the expert escape hatch — each still run through the
Initialize coverage check so even an override is verified *total*, not trusted.

---

## Framework-first implementation: session memory planning and budget selection

This is the next implementation focus after the current policy/accounting stack. Workload-envelope
enforcement and additional bounded fallback coverage remain explicit deferred items below; they
are not prerequisites for building a planner whose result is conditional on declared workloads.
No new kernel fallback or calibrated latency model is required for the first framework slice.

### Responsibility boundary

An EP owns route eligibility, complete workspace recipes, and lowering a selected contract to
runtime dispatch gates. The framework owns allocation identities and lifetimes, session budget
composition, placement, contract selection, and final verification. The framework must not learn
CUDA-specific route names or reproduce kernel scratch formulas.

Expose a small queryable menu of **already enforceable** contracts per candidate node or fused
region. The following is a conceptual internal interface, not a new public API:

```
EnumerateContracts(node_or_region, workload, device, user_constraints)
  -> [
       contract_id,
       admitted phase/domain description,
       allocation requirements with resource/slot ID, device, size bound, alignment, and lifetime,
       bound evidence and unresolved terms,
       runtime coverage or explicit-failure conditions,
       optional measured/model cost or ordinal preference
     ]

EvaluateSessionPlan(placement, selected_contracts, workload, execution_plan, budget)
  -> initialization peak, Run backing peak, uncovered terms, allocation plan

CommitContract(node_or_region, contract_id, plan_generation)
  -> immutable kernel binding used by Level 2 and runtime
```

`Level1MemoryEstimate` is a useful compatibility adapter, but its one runtime-workspace scalar is
not sufficient for lifetime-sensitive selection. Keep per-phase requirements and distinguish
session-persistent prepack, initialization scratch, per-Run backing roots, and node-local scratch.
For the existing GQA consumer, its EP-internal phase masks remain opaque to the framework.

Each contract must include the maximum over **all** routes its runtime selector still permits.
Unavailable analytical terms must remain unavailable, not become zero or inherit a sampled peak
as a proof. Runtime explicit failure is a valid limitation of the first implementation; claiming
successful coverage for every workload state requires stronger eligibility/coverage proof.

Prune dominated candidates only when their entire memory/lifetime requirement vectors and coverage
are comparable. Comparing just workspace bytes can incorrectly discard a contract with a better
persistent-memory or initialization footprint.

### One ledger of physical allocation roots

Build a per-device ledger rather than summing per-node costs. Identify physical allocation roots
and their aliases, owners, locations, and release events; charge each root once. Deduplicate only
sharing established by planner alias metadata or allocation ownership, not matching tensor names
or an assumption that prepacking will reuse storage. The relevant categories are:

| Category | Lifetime and accounting |
|---|---|
| Initializers and packed weights | Include the original device copy until its actual last consumer releases it; packed destinations can coexist with it during Initialize. Shared initializers/prepack roots are charged once. |
| Persistent workspace and capture buffers | Remain live across Runs; a persistent decode slot also overlaps prefill. Capture-retained addresses/pools cannot be released at the end of a host kernel call. |
| KV cache, bound inputs, retained outputs | Include resident backing even when owned by the caller. Use allocation identity to avoid charging past/present aliases twice; retained outputs need an explicit release/retention assumption. |
| Activations and cross-device copies | Producer-to-last-consumer lifetime, extended by asynchronous completion; count GPU-side staging and transfer buffers created by CPU/GPU placement. |
| Node workspace and runtime preparation | Respect per-phase route bounds and declared coexistence. Lazy profiling scratch is separate from steady execution only when its non-overlap is established. |
| Initialization scratch | Overlaps currently resident source/packed weights, constructor allocations, and any other concurrent initialization work. |
| Allocator and EP overhead | Alignment, arena growth/retention, pool fragmentation allowance, handles/library allocations, and explicit reserve outside the managed pools. |

Use `SequentialExecutionPlan` allocation/reuse information, release actions, stream assignment, and
notifications to derive lifetimes. A topological node order or return from `Compute()` is not a GPU
completion event. Reuse storage only when the execution plan and allocator establish happens-before;
otherwise conservatively retain overlapping allocations.

Control-flow runtime scratch uses branch maxima only for mutually exclusive execution, while
weights materialized for both branches remain resident. Loop-carried state and retained scan outputs
must be counted; an unbounded accumulated output makes the corresponding bound unavailable.

### Budget physical backing, not only live tensor bytes

The live-byte sum is a useful diagnostic lower bound for pool sizing, not the final VRAM budget.
Memory-pattern roots and arena regions can remain physically allocated after their values are dead.
Different shapes/phases may also leave multiple cached pools resident. Compute their retained
backing footprint with the actual allocation strategy; do not replace that footprint with the
largest observed live-byte peak.

For fixed placement, contracts, and an explicitly declared number of active Runs:

```
InitPeak(device) = maximum physical resident backing during Initialize
RunPeak(device)  = maximum retained/session and active-Run backing over admitted workloads
SessionPeak     = max(InitPeak, RunPeak) + non-pool overhead/reserve

SessionPeak(device) <= session-associated device budget
```

The session-associated budget covers both session-owned backing and caller-owned backing attributed
to the session. The terms are disjoint: if weights or persistent slots occupy an arena root, they are
included in that root, not added again outside it. For each reusable pool, assign aligned offsets from bounded
lifetimes and use the resulting backing size, including alignment/fragmentation, in the budget.
Reserve is an explicit allowance, not proof that every unknown allocation has been covered.

A backing plan must drive allocation, not just report hypothetical reuse. Execution-frame/kernel
workspace access resolves declared slots to the assigned roots/offsets. If an allocator or kernel
cannot honor that layout, evaluate its actual allocation strategy instead; do not assume the
planned sharing occurred. Undeclared allocation requests need budget-aware charging/rejection,
not an untracked path to the default allocator.

Distinguish the session cap from CUDA's existing `gpu_mem_limit`, which limits the BFC arena rather
than all device allocations. The planner should derive consistent arena/pool caps from the session
budget after accounting for allocations outside those pools; it must not silently reinterpret the
existing provider option.

Report the baseline used: a free-memory snapshot excludes allocations already resident when it is
taken, whereas a full-session budget includes session-owned residents. Reconcile existing/shared
allocations once rather than subtracting them twice. Other processes and shared allocators require
an external reservation/admission mechanism for a device-wide guarantee.

### Planning and commitment order

Planning must finish before bulk device initializer copies, kernel-constructor scratch, and PrePack
can spend the budget. Level-2 verification alone is too late to prevent initialization OOM.

```
Declared workload + device budget + constraints
    -> query capability/contract menus without materializing device kernels
    -> tentative placement, transfers, and execution/allocation plan
    -> lifetime/backing evaluation and contract/placement selection
    -> finalize transformations; invalidate/re-evaluate changed or fused regions
    -> freeze placement, contract IDs, allocation strategy, and plan generation
    -> allocate initializers, create kernels, PrePack
    -> Level-2 verify exact requirements against the selected contracts and global plan
    -> Run using the committed bindings and allocation plan
```

Capability queries must be provisional and side-effect-free with respect to resource commitment.
Layout passes and fusion can change node identities and lifetimes, so a binding belongs to a graph
revision and must be transferred only with validated equivalent metadata; orphaned bindings fail
verification. A fused region requires its own contract, not the sum of constituent scratch peaks.
Use bounded deterministic planning/refinement passes rather than allocating and trying to recover
from OOM.

Store selections with the finalized `SessionState`. Kernels must consume the selected contract ID
and compatible descriptor version, rather than independently re-resolving the session intent to a
different route set after placement. A change to route permissions, lifetime, capture state, or pool
strategy invalidates the plan even when one node's workspace scalar is unchanged.

L2 may tighten a requirement without broadening permissions. A larger requirement needs a verified
new plan before allocation/use or an explicit Initialize error; a warning followed by dynamic growth
cannot satisfy the opt-in hard-budget mode.

**Current reconciliation gaps, not implemented guarantees:** the invariants above are Stage 4/5
requirements, not properties already supplied by the existing post-partition callbacks:

- `SetNodeReplacementCallback` calls `ConsolidateWorkspaceReservations`, which sums source runtime
  reservation bytes and persistent/prepack bytes and takes the maximum initialization scratch.
  This preserves bookkeeping; it neither queries the replacement implementation nor creates its
  contract. A fused node may need more or less than that aggregate.
- `SetNodeRemovalCallback` erases removed-node reservations before finalization checks for orphaned
  entries. The orphan check therefore sees only surviving records; it cannot detect a binding
  obligation already erased by the callback. Track explicit retirement, validated transfer, or
  invalidation/re-estimation so a legitimate removal is distinguished from lost ownership.
- If `DeclareWorkspaceRequirements()` succeeds with an empty list for a node with a nonzero L1
  reservation, current finalization increments `missing_declaration` and continues without setting
  `strict_verification_failed`. Empty/unsupported declarations are not proof of zero workspace.
  The proposed strict mode must reject unresolved nonzero reservations, or require explicit evidence
  of a valid zero-workspace replacement, rather than treating the diagnostic counter as enforcement.

Preserve legacy behavior outside the new opt-in contract/budget mode. Its verification must audit
transformation provenance and unresolved obligations, not merely inspect the final reservation map.

### First budget selector: deterministic, not an optimal-latency claim

Do not move the current greedy CUDA cutoff unchanged into a new class. Budget acceptance is a
whole-plan property: changing one node affects shared weights, the transient peak, and transfer edges.
Keep provider support and user-pinned routes/placement as hard constraints; enable this selector
explicitly so default `auto` behavior and existing named-policy semantics remain unchanged.

Start with a bounded search over the small contract menus:

1. Build the GPU-supported placement with allowed preferred contracts and evaluate both peaks.
2. If it does not fit, trial lower-memory contracts on the peak-driving nodes/regions. Recompute
   the global backing footprint for each trial; use the actual peak reduction, not local bytes.
   Include bounded bundles across tied peaks so zero immediate saving from one change does not
   prematurely force CPU offload.
3. Prefer feasible contract changes to creating CPU boundaries when the current ordinal objective
   favors GPU residency. If contract changes cannot fit, trial supported CPU regions and include
   their changed weight residency, activation copies, and boundary costs.
4. Stop on a verified feasible plan or report that no feasible plan was found within the available
   menus/search limits, with a byte breakdown. Do not claim infeasibility or optimality from a greedy
   search. Optional upgrades consume remaining headroom only after re-evaluation.

Use a deterministic tie-break and explicitly tagged preferences until comparable timing data
exists. For example, assume 9 GiB of common planned backing/reserve and two sequential nodes whose
preferred contracts each need a 2 GiB reusable scratch root. The peak is 11 GiB, not 13 GiB.
Under a 10 GiB budget, switching just one node to a 0.5 GiB contract leaves the peak at 11 GiB;
switching both permits a 0.5 GiB root and a 9.5 GiB plan without CPU offload. This saving depends
on the allocator actually using the shared planned root.

For a calibrated later optimizer:

```
minimize end-to-end cost(placement, contracts, CPU/GPU boundary edges, workload mix)
subject to SessionPeak(device) <= budget(device)
```

The peak is over admitted workloads, not a frequency-weighted average. Phase frequencies can weight
the performance objective only. Correlated phases across nodes must be guaranteed by the workload
contract; otherwise enumerate or conservatively bound mixed per-node states rather than assuming
every node enters decode together.

### Integration and first deliverables

Reuse existing machinery rather than introducing an unrelated executor or allocator:

| Existing surface | Proposed responsibility |
|---|---|
| `resource_accountant.h/.cc`, partitioner bookkeeping | Preserve local estimate provenance and compatibility; introduce whole-plan evaluation alongside the additive baseline. |
| `TransformGraph` replacement/removal callbacks | Replace source-sum-only reconciliation with target-recipe requirements; explicitly record retirement/transfer/invalidation so erased entries cannot hide unresolved bindings. |
| `level1_memory_estimate.h`, EP estimators | Adapt existing scalar estimates; participating EPs additionally expose contract-specific lifetime requirements. |
| `allocation_planner.cc`, `sequential_execution_plan.h` | Bound value sizes, deduplicate physical roots, derive release/stream constraints, and support provisional plans. |
| `mem_pattern_planner.h`, `execution_frame.cc`, `bfc_arena.cc` | Translate liveness into aligned retained backing consistent with actual allocation and growth/reuse behavior. |
| `session_state.cc`, `DeclareWorkspaceRequirements()` | Persist selected bindings/plans and reconcile L2 requirements, ownership, and lifetimes. Close the current strict-verification gap for empty declarations with nonzero reservations; distinguish proven zero from unsupported/missing coverage. |
| CUDA `GetCapability()` | Report support and candidate contracts; let the opt-in framework selector own global acceptance instead of stopping at the first additive threshold. |

Split the work into reviewable PRs:

1. **Session GPU lifetime/backing report:** fixed placement and contracts, initialization plus Run
   accounting, identity/alias deduplication, and explicit unknowns. No default placement changes.
2. **Enforceable session budget plan:** pre-allocation feasibility gate, aligned reusable backing,
   reserve/pool consistency, and L2 reconciliation. Begin with one active Run and a validated
   single-stream plan, backed by caller-guaranteed serialization or framework Run admission;
   model unsupported execution modes conservatively or reject the budget mode.
3. **Budget-constrained contract selection:** generic EP contract menu and immutable per-node
   bindings, with the deterministic selector above. Begin with a GQA adapter using #33316's existing
   enforced masks, then add other kernels only when their committed routes are enforceable.
4. **Placement selection:** add CPU-region alternatives and transfer-edge costs to the same planner,
   then expand control flow, concurrent Runs, and shared-session/device admission.

Per-Run workspace is reusable across sequential nodes, not automatically across concurrent Runs.
An isolated one-Run report is not a shared-device guarantee. CUDA graph capture needs a declared
retained-buffer model; unsupported capture modes must remain uncovered, not silently use the
non-capture estimate.

Targeted tests should cover sequential scratch reuse, asynchronous overlap, shared weights and KV
aliases, source/destination prepack coexistence, persistent decode storage during prefill, allocator
alignment/retention, initialization-dominated peaks, phase correlations, changed/fused bindings,
replacement requirements larger than source sums, removed bindings erased before orphan checking,
empty L2 declarations with nonzero reservations, proven-zero replacements, and a contract change
that avoids CPU offload. Synthetic framework plans can exercise these before GPU integration;
observed device peaks subsequently validate the model rather than replace it.

### Deferred correctness/coverage TODOs

- [ ] **Workload-envelope enforcement/admission:** add optional ORT-side checks where caller
  guarantees or existing validation do not suffice, or admission/rebudgeting before oversized
  workloads execute. Query/chunk, batch, active KV, allocated cache, and concurrency bounds are
  distinct; windowed caches do not require accumulated context to stay below resident capacity.
- [ ] **Bounded fallback coverage:** add low-workspace fallback implementations or stronger
  eligibility coverage so restrictive phase contracts can serve more states without an unaccounted
  escape. Existing explicit failure remains the policy for uncovered runtime availability.

These TODOs are deferred, not erased. Until workload, allocator/runtime, and ownership assumptions
are guaranteed, publish the framework result as a **conditional session budget plan**, not an
unconditional no-OOM promise. Caller admission guarantees can satisfy the workload condition without
duplicating them inside every kernel.

---

## Recommended roadmap

```
Stage 0 (baseline): precise known initializer/output bytes + generic 50% workspace heuristic.
                    Neither route-aware nor a proof bound.

Stage 1: Unify the runtime allocation with the existing estimator recipe system.
         - The recipe system (group_query_attention_workspace*.{h,cc}) already exists and
           drives the L1 estimator; migrate ComputeInternal to allocate from the same
           GetGQA*WorkspaceRecipe builders instead of sizing scratch inline.
         - Route-specific sizing formulas agree by construction; L1 remains their envelope max.

Stage 2: Static route predicate + route-aware, envelope-driven estimate.
         - Extract seq-free flash/MEA eligibility into a pure helper.
         - W_L1 = max over admitted (phase × route) of CompleteWorkspace.
         - If a reachable route has no proof bound, fail closed for capacity-aware placement.
         - Reclaims most VRAM via static analysis (Case A models).

Stage 3: Committed policy + enforcement.
         - Resolve GQA auto/latency/memory/safe intents to phase-aware route masks.
         - Share that resolver across L1, L2, and runtime.
         - Runtime classifies prefill, decode, or chunked prefill and may fall back only inside
           that phase's committed mask; otherwise it fails explicitly.
         - Explicit failure is retained for uncovered runtime availability.
         - Additional low-workspace fallback coverage is a deferred TODO, not a framework blocker.

Stage 4 (next): Whole-session GPU lifetime/backing accounting and a budget plan.
               - Account Initialize as well as Run, including physical pool retention and reserve.
               - Deduplicate shared/aliased roots; reuse scratch only under proven non-overlap.
               - Reuse existing execution/allocation planners and reconcile L2 before Run.
               - Audit reservation retirement/transfer through transformations; final-map orphan
                 checks alone miss obligations erased by removal callbacks.
               - Fail closed on unresolved nonzero reservations with empty/missing L2 declarations;
                 the current missing_declaration counter does not itself enforce strict failure.
               - Report conditional assumptions and unavailable terms explicitly.

Stage 5: Budget-constrained contract and placement selection.
         - Enumerate EP-owned enforceable contracts and persist the selected bindings.
         - Bind fused/replaced implementations to their own requirements and graph revisions;
           source reservation sums are bookkeeping, not replacement contracts.
         - Recompute whole-session backing for each contract/CPU-region alternative.
         - Begin with deterministic preferences, without claiming optimal latency.

Stage 6 (later): Envelope profiling, comparable cost calibration, and adaptive policy tuning.
                 - Exercise relevant shape/route regimes, including chunked prefill.
                 - Treat observed allocation above its bound as a correctness failure.
                 - Never substitute the sampled peak for a proof bound in strict placement.

Deferred TODOs: optional workload-envelope enforcement/admission and bounded fallback coverage.
```

The proof remains route-local until Stage 4 accounts for global lifetimes, backing, and reserve.
A session-wide guarantee additionally requires workload bounds and ownership/concurrency assumptions
to be guaranteed, either by ORT or the caller's admission contract. Neither a route-local bound nor
an empirical peak is a complete no-OOM guarantee.

### Actionable now vs. forward-looking

- **Existing foundation:** route-aware L1 accounting and #33316's opt-in phase-aware GQA commitment
  with `W_L1` closed over the permitted set. Default behavior remains unchanged.
- **Next focus:** Stage 4's whole-session GPU backing report/budget plan and Stage 5's deterministic
  contract/placement selector. Calibrated latency and new fallback implementations need not block them.
- **Forward-looking:** calibrated end-to-end comparison including CPU-boundary costs and adaptive
  tuning. Keep cost provenance in the contract interface so this can improve selection later.
- **Global dependency:** a local estimate reduction helps only when it reduces the session's
  peak-driving backing. Tightening a node estimate does not by itself prove the whole-session peak.
- **The hard prerequisite for any Case B reduction:** runtime enforcement must land with the
  estimate that removes a currently reachable route. The GQA-first implementation gates every
  dispatch branch with the committed mask and fails when no permitted route serves the Run. A
  future low-workspace fallback can make more restrictive masks total without re-admitting the
  quadratic route.

---

## Key insight

Do not force one static estimate to be both safe and tight, and do not estimate the `max` over every
implementation in the source tree. Instead:

1. **Commit to a policy** — a total, enforceable `P: state → non-empty set<route>`. The estimate is
   `max over admitted states x and permitted routes r ∈ P(x) of CompleteWorkspace(x, r)`.
2. **Bind per `(node, phase)`**, not per node — because one kernel serves prefill and decode with
   different routes, and preference is phase/shape-dependent (a scalar `speed_rank` is insufficient).
3. **Excluding a route from the estimate is earned, not assumed** — either *prove* it unreachable
   (Case A: seq-free flash/MEA eligibility already forecloses unfused) or *enforce* it (Case B:
   bounded fallback / error). Estimation and execution are one commitment.
4. **L1 never detects the phase** — it enumerates envelope-admitted phases and takes the `max` over
   each phase's permitted routes, which is why `P` must be statically resolvable at partition time.
5. **Profiles are evidence, not proof** — use empirical peaks to detect estimator bugs and calibrate
   cost models, never to replace an analytical upper bound required for strict placement.
6. **Per-node proof is not global proof** — whole-graph liveness and allocator reserve determine the
   session peak. The next focus is framework lifetime/backing accounting followed by
   budget-constrained contract and placement selection.

Bounded fallback coverage broadens the admitted workload, while envelope enforcement or caller
admission establishes its bounds. Neither is a prerequisite for developing a framework planner
that reports its assumptions honestly and never escapes the selected route contracts.
