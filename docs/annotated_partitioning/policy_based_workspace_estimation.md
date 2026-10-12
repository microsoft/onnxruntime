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
current runtime priority order. Empirical timing can replace that ordering only after it is
collected over the same enforced workload envelope.

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

**The bounded fallback is a precondition for a total policy** — it is the ORT analog of cuDNN's
guaranteed zero-workspace algorithm, which is *why* cuDNN/TensorRT can safely invert to
"budget → route." ORT's current fallback is the *expensive* route, which is the inversion of what
budget-driven selection needs.

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
`total_sequence_length` resides on the GPU and the helper cannot distinguish the first prompt from
a later chunk, the existing runtime behavior reports the multi-token step as non-first, so policy
enforcement conservatively uses the chunked-prefill mask. L1 has already admitted and sized both
general phases.

Everything L1 needs is static:

| Ingredient | Source |
|---|---|
| which phases are admissible | graph structure: node has `past_key`/`present_key` edges ⇒ autoregressive ⇒ `{prefill, chunk, decode}`; no `past` input ⇒ prefill-only |
| query bound `P` | `max_shape_override` on `input_ids` |
| KV capacity `C` | overridden `past_key` seq dim / GQA capacity knob |
| routes per phase `P(p)` | the committed policy, evaluated symbolically |
| route eligibility | seq-free static predicates (dtype, device_prop, head_size, heads, flags) |

### `max`, not `sum` — and the one caveat

In ORT today a single Run is **single-phase**: GQA rejects mixed regimes with
`batch_size must be 1 when sequence_length > 1 and past context is given`
(`group_query_attention_helper.h`). Phases occur in *different* Runs and the arena is reused across
Runs, so the reservation is the **max across phases**, not the sum — dominated by the linear prefill
term.

> **Caveat:** if ORT ever adopts vLLM-style **continuous batching / mixed chunked-prefill + decode
> in one forward** (ragged batch), prefill and decode routes become simultaneously live in one Run,
> and that Run's estimate must **sum** the coexisting `(phase → route)` workspaces. The batch=1
> guard is what currently protects the `max` model.

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

**`latency` is the zero-config default and the right choice for the common case** — a single
interactive session whose model comfortably fits in VRAM. The reason the other two intents exist is
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
  A system-wide no-OOM guarantee additionally requires runtime enforcement of the query/KV
  envelope, whole-graph peak/lifetime accounting, and explicit allocator/runtime reserve; those
  are not supplied by a route policy alone. `safe` is also useful for reproducibility, debugging,
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
(from the GQA knob). Until runtime rejects query or KV values beyond those declarations, this is an
estimation contract rather than a runtime-enforced no-OOM contract.

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
may sit at the switch boundary, not the envelope corner). Stage-4 profiling *validates* this bound; it
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

**Honest dependency:** comparability needs latency numbers as trustworthy as the workspace numbers,
so it realistically lands *after* the Stage-4 envelope-profiling calibration; before that, latency
ranks are heuristic. The recommendation is to **design the `(W_L1, cost)` interface now** — the
estimator emits a cost alongside each policy's workspace — even though the optimizer that consumes it
arrives later, so the estimator is not reworked.

### Automatic selection is this comparison, run for the user

The Tier-0 end state is this global comparison solved on the user's behalf: the user
declares **budget + workload envelope + intent (`latency`/`memory`)**, and the framework enumerates
the **kernel author's** policy menu, evaluates each policy's `(W_L1, cost)` for their device and
envelope, and picks the placement+policy that meets budget with the best objective. The user never
authors a dispatch table; the kernel author owns the menu, the framework owns the selection, and
explicit `fused_only`-style overrides remain the expert escape hatch — each still run through the
Initialize coverage check so even an override is verified *total*, not trusted.

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
         - Follow-up: add a low-workspace fallback so stricter policies can remain total in Case B.

Stage 4: Envelope profiling Run to validate and calibrate.
         - Exercise every relevant shape/route regime, including chunked prefill.
         - Treat any observed allocation above W_L1 as a correctness failure.
         - Never substitute the sampled peak for a proof bound in strict capacity placement.

Stage 5: Whole-graph peak/lifetime accounting.
         - Combine persistent residents, activation liveness, overlapping workspaces, and reserve.
         - Reuse sequential scratch instead of summing every per-node maximum.

Stage 6: Adaptive policy tuning from observed traffic, constrained by proven contracts.
```

The proof remains route-local until the workload envelope is runtime-enforced and Stage 5 accounts
for global lifetimes and reserve. Before both land, neither a route-local bound nor an empirical peak
is a complete no-OOM guarantee.

### Actionable now vs. forward-looking

- **Actionable now** (opt-in flag, no default change): the contract triple with `W_L1` closed over
  the permitted set, the GQA-first `latency`/`memory`/`safe` resolver shared by L1/L2/runtime, and
  the L1/L2 requirement-vs-strategy split with the lifetime caveat.
- **Forward-looking** (gate on the Stage-4 profiling that makes latency comparable): full policy
  comparability including CPU-boundary cost, and the auto-optimizer that consumes it — but ship the
  `(W_L1, cost)` interface early so it drops in without rework.
- **Separate global dependency:** Stage 5 consumes proven per-node bounds but owns activation,
  workspace, persistent-buffer, and allocator lifetimes. Tightening a node estimate does not by
  itself prove the whole-session peak.
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
   session peak and remain separate accounting work.

The bounded fallback is the linchpin: it is what lets a policy be total (Case B) and makes the
route-level budget decision in the constrain-then-select model (cuDNN/TensorRT) sound.
