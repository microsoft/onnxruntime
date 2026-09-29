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

1. **Safety** — the estimate must be an upper bound, so a placement decision never causes OOM.
2. **Tightness** — the estimate must be close to actual, so we do not waste VRAM and evict
   heavy nodes to CPU that would have fit.

Forcing a *single static number* to be simultaneously a safe upper bound and a tight value is
impossible for a dynamic-shape op whose kernel can take several routes with very different
footprints. "Conservative" wins, VRAM is wasted, and utilization drops.

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
- **Dispatch policy `π`:** a **committed, total, enforceable** function `π: x → route` whose
  **image is a restricted route set** (deliberately excluding the expensive fallback). It is *not*
  a description of today's open-ended `if`-ladder; it is a contract that the runtime is modified
  to obey.

### The L1 requirement under a policy

```
W_L1(node) = max over admitted runtime states x:  CompleteWorkspace(x, π(x))
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

### The three properties that make `π` the estimation object

1. **Total over the admitted workload.** `π(x)` must be defined for every admitted `x`. If some
   admitted state maps nowhere (and the fallback is prohibited), the policy is **invalid** — narrow
   the admitted set or add a bounded fallback. This is the coverage condition.
2. **Bounded image excluding the expensive route.** The set `{π(x)}` is small, known, and excludes
   the quadratic unfused route *by construction*.
3. **Enforceable — estimation and execution are one commitment.** The runtime must dispatch to
   `π(x)`. A would-be out-of-policy route (the ladder wanting to fall to unfused) is a *policy
   violation* → error or bounded fallback, **never** a silent quadratic allocation.

> **Soundness invariant:** an L1 estimate is sound only if it bounds *every route the runtime can
> actually execute*. A route may be dropped from the `max` **iff** the runtime is incapable of
> running it. Excluding unfused from the estimate while the runtime can still fall to unfused is
> *unsound* and will OOM on exactly the Run where unfused fires.

### The contract: bind constraints, not a kernel

The unit of commitment is not a chosen kernel but a **contract** — the triple:

```
contract(node) = ( admitted workload = envelope,
                    permitted routes  = image(π),
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
never executes in the current source, with no code change.** `max(prefill, decode, chunk)` is sound
as-is. This is the common production case (fp16/bf16, head_size 64/128, SM80+, no bias).

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
- Only after that change is unfused out of `image(π)`, and only then may the `max` drop it.

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
which is why the benchmark added an out-of-band knob
(`ep.cuda.gqa_workspace_max_total_sequence_length`) to hand the estimator the KV bound directly.
The invariant is `S_q ≤ total_sequence_length ≤ kv_cache_capacity`; the two axes coincide only in
the first full prefill.

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
W_L1 = max over p ∈ AdmittedPhases:  CompleteWorkspace( corner(p, envelope), π(p) )
```

Everything L1 needs is static:

| Ingredient | Source |
|---|---|
| which phases are admissible | graph structure: node has `past_key`/`present_key` edges ⇒ autoregressive ⇒ `{prefill, chunk, decode}`; no `past` input ⇒ prefill-only |
| query bound `P` | `max_shape_override` on `input_ids` |
| KV capacity `C` | overridden `past_key` seq dim / GQA capacity knob |
| route per phase `π(p)` | the committed policy, evaluated symbolically |
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
  (`include/onnxruntime/core/framework/resource_accountant.h`) distinguishes
  `kFallback / kEstimator / kProfile`; route-aware estimation moves GQA from a fallback multiplier
  to a trustworthy `kEstimator` value.
- **L2 (Initialize):** `DeclareWorkspaceRequirements` declares the concrete per-slot need for the
  chosen envelope. `session.strict_workspace_verification` governs whether an L2 declaration larger
  than the L1 reservation fails Initialize (`=1`, strict) or logs and retains dynamic allocation
  (`=0`, default).
- **Single-source the sizing — the estimator half already exists.** The sizing is already
  consolidated on the estimator side: the recipe system in
  `group_query_attention_workspace*.{h,cc}` (`GetGQAPreparationRecipe` plus per-backend
  `GetGQA{Xqa,Flash,MemoryEfficient,Unfused}WorkspaceRecipe`, composed by
  `GetGQACompleteWorkspaceRecipe`, overflow-checked and validated) is the single source that drives
  `EstimateGroupQueryAttentionWorkspace` at partition time. What is **not** yet unified is the
  runtime: `ComputeInternal` still sizes its scratch inline, and `GQABufferRequirements::Compute`
  (`group_query_attention_impl.h`) sizes only `qkv_buffer`. So the remaining work is
  one-directional — **migrate the runtime allocation to consume the same recipe builders the
  estimator already uses** — after which estimate == allocation by construction and profiling
  becomes *validation*, not a *requirement*. (The recipe layouts are marginally larger than the
  current inline math — e.g. the unfused recipe 256-aligns its QK and softmax regions separately —
  so this is a sound behavior reconciliation, validated by `group_query_attention_workspace*_test.cc`
  plus the GQA op tests, not a byte-identical refactor.)

### Requirements vs. allocation strategy — and the lifetime invariant

L1 answers **how much** (the peak-concurrent requirement of the whole policy); L2 answers **how it is
provided** (a persistently preallocated slot vs. transient scratch/arena). These must stay separable
so allocation strategy can be tuned by benchmark without moving the placement number — the benchmark
lesson (do *not* persist prefill-dominated GQA; *do* persist phase-invariant MatMulNBits) is an L2
decision that should not perturb L1.

The invariant that keeps them separable:

> **L2 may vary a buffer's *source* freely without touching L1, but changing its *lifetime* changes
> L1's concurrency.** A persistently preallocated decode buffer is live *during* prefill, so it
> stacks with prefill scratch — L1 must then budget `prefill_scratch + persistent_decode_slot`, not
> `max(prefill_scratch, decode_slot)`. If L2 changes lifetimes, L1 re-budgets under the declared
> lifetime model.

`strict_workspace_verification` remains the guardrail that an L2 declaration never exceeds the L1
reservation.

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

**Tier 0 — auto (default).** Framework synthesizes `π` from `(device caps, envelope, budget)`:
pick the fastest route set it can *prove* total over the admitted phases; else degrade to the
bounded/safe policy. No user action.

**Tier 1 — intent-level named policy (portable session option).**

```
session.attention_dispatch_policy = "latency" | "memory" | "safe"
```

Intent, not mechanism — portable across ops/EPs. `latency` favors high-workspace fast routes;
`memory` favors low-workspace routes (more nodes fit on GPU → less offloading); `safe` forces the
bounded fallback everywhere. Belongs in **session options** (hardware-neutral).

**Tier 2 — explicit route menu (EP-specific provider option, expert).**

```
ep.cuda.gqa_dispatch_policy = "prefill=flash;decode=xqa;fallback=error"
```

The literal `π`, for benchmarking / power users. EP-scoped because routes are hardware-specific —
consistent with the existing `ep.cuda.gqa_workspace_max_total_sequence_length`. Today's scattered
per-route toggles become the **compile target** this policy lowers to:

- `ORT_ENABLE_XQA`, `ORT_DISABLE_FLASH_DECODE` — environment variables
  (`group_query_attention.cc`)
- `AttentionKernelOptions` via `GetAttentionKernelOptions()` — session-scoped
  (`UseFlashAttention()`, `UseEfficientAttention()`, `UseCudnnFlashAttention()`)
- model attributes (`num_heads`, `local_window_size`, `softcap`, …)

### Validation is non-negotiable (regardless of who chose)

At Initialize, run a **coverage check**: is `π` total over the admitted `(phase × envelope)` on
*this* device? Reuse the strictness pattern already modeled by
`session.strict_workspace_verification`:

- **strict = 1** → invalid policy (e.g. `decode=xqa` on SM70 where XQA is unsupported) **fails
  Initialize** with a precise diagnostic. No silent OOM.
- **strict = 0** → log a warning and **degrade** the offending phase to the bounded fallback.

Explicit choice ≠ unchecked choice.

### Orthogonal knobs and scope

- **Budget is separate from policy** (cuDNN workspace-limit / vLLM `gpu-memory-utilization`
  precedent): the user supplies the VRAM budget/reserve; the policy defines the route *function*;
  ORT checks `W_L1(π) ≤ budget`.
- **Session-scoped by default.** All attention nodes share one KV-cache envelope, so `π` is one
  function for all GQA nodes (matching `AttentionKernelOptions`), with an optional per-node
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

ORT's current per-kernel "report your worst-case size" does **neither**: it estimates analytically
(inherits estimation error) but reports the **worst route** (unfused). This design adopts both: a
**static route-map + policy** for the plan (constrain-then-select), and an **envelope profiling
Run** to calibrate (pin-then-measure), with the **bounded fallback** as the linchpin that makes
budget-driven selection safe. Heed vLLM's documented pitfall: profile the *true* worst envelope
(include the chunked-prefill shape), or the measured peak under-predicts and OOMs.

---

## Worked example: GQA policy `π_GQA`

```
Policy π_GQA(x):
  prefill  (is_first_prompt)          → Flash        if flash_supported   else INVALID
  decode   (seq==1 && !is_first)      → XQA          if xqa_supported
                                         else Flash   if flash_supported
                                         else INVALID
  chunk    (1 < seq < total)          → Flash        if flash_supported   else INVALID
  # unfused ∉ image(π_GQA)  →  prohibited (error or bounded fallback)
```

Envelope: query bound `P` (from `max_shape_override`), KV capacity `C` (from the GQA knob).

```
prefill/chunk :  CompleteWorkspace(S_q=P, S_kv=C, Flash) = lse + lse_accum + out_accum   [linear]
decode        :  CompleteWorkspace(S_q=1, S_kv=C, XQA)   = GetXQAScratchSize(C)          [small]
W_L1 = max(...) = flash prefill term        ← linear, ~38x below the unfused reservation
```

Using `(S_q=P, S_kv=C)` as a single safe corner slightly over-approximates any individual Run
(first prefill has `S_kv=S_q`; a deep chunk has small `S_q`) but is a valid, still-linear upper
bound. Tighten by tracking the per-phase `(S_q, S_kv)` coupling if needed. If a policy branch
switches route at an interior shape threshold, take the `max` **piecewise per route-region** (the
max may sit at the switch boundary, not the envelope corner).

---

## Comparability: one contract → a Pareto set of contracts

The design's natural end-state elevates each node from *one* contract to a small **Pareto set** of
contracts, each exposing a `(W_L1, latency)` pair. Placement then solves a **global** objective —
minimize model latency subject to `Σ resident + peak workspace ≤ capacity` — instead of greedily
fitting each node independently.

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

Tier-0 auto-selection is exactly this global comparison solved on the user's behalf: the user
declares **budget + workload envelope + intent (`latency`/`memory`)**, and the framework enumerates
the **kernel author's** policy menu, evaluates each policy's `(W_L1, cost)` for their device and
envelope, and picks the placement+policy that meets budget with the best objective. The user never
authors a dispatch table; the kernel author owns the menu, the framework owns the selection, and
explicit `fused_only`-style overrides remain the expert escape hatch — each still run through the
Initialize coverage check so even an override is verified *total*, not trusted.

---

## Recommended roadmap

```
Stage 0 (today): precise resident + conservative (unfused-bounded) workspace.
                 Safe, wasteful — the current state.

Stage 1: Unify the runtime allocation with the existing estimator recipe system.
         - The recipe system (group_query_attention_workspace*.{h,cc}) already exists and
           drives the L1 estimator; migrate ComputeInternal to allocate from the same
           GetGQA*WorkspaceRecipe builders instead of sizing scratch inline.
         - Estimate == allocation by construction. Highest leverage.

Stage 2: Static route predicate + route-aware, envelope-driven estimate.
         - Extract seq-free flash/MEA eligibility into a pure helper.
         - W_L1 = max over admitted (phase × route) of CompleteWorkspace.
         - Reclaims most VRAM via static analysis (Case A models).

Stage 3: Committed policy + bounded fallback + enforcement.
         - Named policies (Tier 0/1/2), coverage validation at Initialize.
         - Bounded fallback route so every policy can be made total (Case B).
         - Runtime prohibits unfused under a committed policy.

Stage 4: Envelope profiling Run to calibrate the estimate.
         - Profile at the max_shape_override envelope (incl. chunked-prefill shape).
         - Replace estimate with measured peak; feed IResourceAccountant.

Stage 5: Adaptive re-partition from observed peaks across real traffic.
```

Each stage strictly increases utilization without reintroducing OOM risk, because the no-OOM
guarantee rests on the **precise resident budget + reserve + enforced bounded fallback**, not on the
workspace estimate being both safe and tight.

### Actionable now vs. forward-looking

- **Actionable now** (opt-in flag, no default change): the contract triple with `W_L1` closed over
  the permitted set (Stages 1–2), the L1/L2 requirement-vs-strategy split with the lifetime caveat,
  and basic auto-selection over a *validated* menu.
- **Forward-looking** (gate on the Stage-4 profiling that makes latency comparable): full policy
  comparability including CPU-boundary cost, and the auto-optimizer that consumes it — but ship the
  `(W_L1, cost)` interface early so it drops in without rework.
- **Single hard prerequisite for all of it:** the dispatch-ladder enforcement change — redirecting
  the unfused terminal (`data.use_unfused = true` in `group_query_attention.cc`) under a committed
  contract — landed as its own small, well-tested PR *before* any smaller estimate is trusted.

---

## Key insight

Do not force one static estimate to be both safe and tight, and do not estimate the `max` over every
implementation in the source tree. Instead:

1. **Commit to a policy** — a total, enforceable `π: state → route` whose image excludes the
   expensive route. The estimate is `max over admitted states of CompleteWorkspace(x, π(x))`.
2. **Bind per `(node, phase)`**, not per node — because one kernel serves prefill and decode with
   different routes, and preference is phase/shape-dependent (a scalar `speed_rank` is insufficient).
3. **Excluding a route from the estimate is earned, not assumed** — either *prove* it unreachable
   (Case A: seq-free flash/MEA eligibility already forecloses unfused) or *enforce* it (Case B:
   bounded fallback / error). Estimation and execution are one commitment.
4. **L1 never detects the phase** — it enumerates envelope-admitted phases and takes the `max` of
   the policy-selected route at each, which is why `π` must be a static, total function evaluable at
   partition time.

The bounded fallback is the linchpin: it is what lets a policy be total (Case B) and what makes the
budget-driven, constrain-then-select model (cuDNN/TensorRT) safe in ORT.
