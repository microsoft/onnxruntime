# K1: Declared State for CPU/GPU Placement

K1 gives ORT the state requirements it needs **before allocating KV buffers**. The state owner produces a
declaration; ORT resolves it against the graph and plans placement; the owner reads the resulting locations and
allocates there. The owner does not have to predict the CPU/GPU cut.

This is the declared-state workstream in the [region-aware placement roadmap](https://github.com/microsoft/onnxruntime/pull/33372)
and a follow-up to the [S0 capability-planning protocol](capability_planning_interfaces.md).
The contract is ownership-neutral: changing from GenAI-owned KV to ORT-owned KV does not change the planner input.

## Draft scope and implementation status

| Implemented in this draft | Not implemented here |
|:--------------------------|:---------------------|
| Strict versioned JSON parser and normalized internal state records | A registered session option, support-negotiation API, or automatic parsing during session creation |
| Literal names and `{layer}` template expansion | GenAI, harness, or model-package producers |
| Dense, unquantized BNSH `com.microsoft::GroupQueryAttention` past/present pairs at graph boundaries | Recurrent-state adapters, paged/quantized caches, internal state edges, or multi-consumer state |
| Graph-derived dtype, KV heads, head size, window mode, and checked per-tensor byte bounds | EP eligibility, estimator adaptation, physical-backing accounting, transfer pricing, or budget admission |
| Capacity and alias projection into an unpublished S0 query | CPU/GPU cut selection, placement commitment, per-cache allocators, or bind-time enforcement |
| Atomic rejection of malformed, conflicting, missing, or unsupported declarations | KV allocation, movement, eviction, rewind, or ownership transfer |

These are opt-in **internal helpers**, like the initial S0 types. They are not wired into `InferenceSession`.
No existing session, legacy resource accountant, EP callback, C API, or KV owner changes behavior.
A successful parse or resolution is **not** an admitted memory plan.

The internal entry points in
[`declared_state.h`](../../onnxruntime/core/framework/declared_state.h) are
`ParseDeclaredStateContract`, `ResolveDeclaredStateContract`, and `ProjectDeclaredStateToPlanningQuery`.
The last helper modifies only an unpublished query; publish it as an owned const snapshot afterward.

## Who produces K1?

The producer is the component that will allocate/manage the buffers, acting on the caller's workload bounds.
The exporter may supply static cache metadata, but the caller must supply the intended session envelope.

| Owner | Proposed producer and source |
|:------|:-----------------------------|
| GenAI-owned KV | `Model::CreateSessionOptions()` translates `genai_config.json` names/layers and the caller's envelope |
| External generation harness | Harness translates CLI/configuration values into the same declaration |
| ORT-managed KV | ORT's chain/session-creation glue translates package `kv_caches` specifications and bindings plus the caller's envelope, before creating each step's session |

GenAI does not read the declaration back. It reads the **resulting plan** after session initialization.
An ORT-managed owner uses that same information internally. A manager with one allocator for all caches must
be extended to select an allocator from each cache's planned location; choosing a single `cache_provider` is
not enough for a mixed CPU/GPU plan.

These producers are follow-ups. This draft does not depend on a particular model-package or stateful-execution
implementation. In particular, a package's `initial_capacity` is not a maximum bound, and an update mode named
`share` must not be interpreted as an in-place past/present alias: adopting an output is different from binding
the input and output to identical storage.

## Simplified end-to-end workflow

The complete workflow below is the intended integration, **not behavior already enabled by this PR**.

```text
BEFORE SESSION CREATION
  1 DECLARE    owner -> ORT
               K1: state names/layers, capacity, alias, lifetime, location policy
               + session envelope: batch, absolute sequence length, live instances

DURING SESSION CREATION
  2 RESOLVE    ORT
               parse K1; match graph state; derive dtype/layout/dimensions/window;
               reject missing, ambiguous, conflicting, or unsupported state
  3 PLAN       ORT
               for each candidate OP/subgraph placement:
                 KV location = planned consumer's input-memory location
                 M2 identifies backing and attributes KV bytes to that device
                 R3 prices required transfers; I1 evaluates budget feasibility
               I3 selects a feasible plan; I2 rechecks before commitment/allocation
               record each past/present tensor's committed location

AFTER SESSION CREATION
  4 QUERY      owner <- ORT
               SessionGetMemoryInfoForInputs / SessionGetMemoryInfoForOutputs
               -> planned location per state tensor
  5 ALLOCATE   owner
               enforce the admitted envelope and live-instance limit;
               allocate each cache on its planned location;
               bind past == present only when alias = past_present
  6 VALIDATE   ORT, before execution
               reject dtype, dimensions, capacity, alias, or location mismatches;
               then Run, without an unplanned state copy or capacity expansion
```

Steps 1-3 need declared numbers and graph metadata, not owner-allocated buffers. There is no circular wait for
GenAI allocation. M2 **reports/attributes memory**; I1 checks budgets; I3 chooses among feasible placements.
Provisional candidate locations are not returned to the owner as if they were committed.

The default policy is `follow_consumer`. For GQA cache ports this means CPU KV for CPU attention and GPU KV for
GPU attention. More generally, honor the selected kernel's input-memory requirements, not just its EP label.
CPU-layer KV consumes host budget, not VRAM. Activation transfers between layers still need R3 accounting.

After initialization, query **both** input and output locations. An identical-storage pair requires compatible
device, device ID, and memory type for both ports. If they disagree, fail rather than binding one buffer and
allowing ORT's ordinary I/O staging path to hide a transfer. Distinct-storage pairs require two allocations on
their respective planned locations. The K3 integration must ensure the location APIs describe the admitted plan;
the APIs' existence alone does not establish K1 or budget-admission support.

A storage pin conflicting with the consumer is not v1. A future K6 contract must explicitly describe and price
host-storage/GPU-compute copy-through. This draft rejects other location policies instead of silently staging.

## Version 1 JSON

Each entry describes one **input/output state tensor pair**, not both K and V. Declare K and V separately.
Templates compactly describe multiple layers; the resolver records exact names and the actual consumer node.

```json
{
  "version": 1,
  "max_batch_size": 1,
  "max_sequence_length": 4096,
  "max_live_instances": 1,
  "states": [
    {
      "id": "key",
      "kind": "kv_cache",
      "input": "past_key_values.{layer}.key",
      "output": "present.{layer}.key",
      "layers": [0, 1],
      "capacity": 4096,
      "alias": "past_present",
      "lifetime": "per_generator",
      "location": "follow_consumer"
    },
    {
      "id": "value",
      "kind": "kv_cache",
      "input": "past_key_values.{layer}.value",
      "output": "present.{layer}.value",
      "layers": [0, 1],
      "capacity": 4096,
      "alias": "past_present",
      "lifetime": "per_generator",
      "location": "follow_consumer"
    }
  ]
}
```

| Field | Contract |
|:------|:---------|
| `version` | Required integer `1`. Unsupported versions fail explicitly. |
| `max_batch_size` | Positive maximum batch dimension of each state instance, including any beam expansion. |
| `max_sequence_length` | Positive bound on absolute total sequence length, including past plus new tokens; not the retained sliding-window size. |
| `max_live_instances` | Positive maximum simultaneously resident per-generator state sets for this session. Serialized Runs do not imply only one resident Generator. |
| `id` | Nonempty, unique declaration-group ID. A resolved tensor is identified by its group and layer; graph names remain the storage references. |
| `kind` | `kv_cache` in v1. Other state kinds need graph adapters before they can be admitted. |
| `input`, `output` | Nonempty literal graph names or names with exactly one `{layer}` token. No regex/glob interpretation. |
| `layers` | Nonempty list of distinct nonnegative integers. Literal names work with a single layer. Expanded names must be globally distinct. |
| `capacity` | Positive upper bound on the pair's physical sequence dimension, in tokens, not bytes or initial allocation size. |
| `alias` | `past_present`: input/output have identical base storage, with no offset. `none`: they have disjoint storage. Unknown is not treated as sharing. |
| `window` | Optional nonnegative assertion against graph `local_window_size` (`0` means disabled). Omission derives it from the graph. The graph's `sliding_window_cache` separately determines whether storage is window-sized. |
| `lifetime` | `per_generator` or `per_session`. The former is multiplied by the declared live-instance bound; the latter has one state set per session. |
| `location` | Required `follow_consumer` in v1; this is a policy, not a CPU/GPU device chosen by the owner. |

All fields except `window` are required. Unknown fields, duplicate JSON object keys, duplicate IDs/layers/names,
invalid enums, fractional/out-of-range integers, and nonpositive envelope/capacity bounds fail.
The parser returns `INVALID_ARGUMENT` for malformed declarations and `NOT_IMPLEMENTED` for unsupported versions,
state kinds, or location policies. Every helper preserves its output on failure.

No dtype or full-shape assertion is needed from an owner that has no graph. The resolver derives element type
from the graph ports, KV heads from GQA, head size from cache shapes, and BNSH layout from the operator contract.
Version 1 supports FP32, FP16, and BF16 caches. Unresolved rank/head size or unsupported element types fail closed.
Optional owner-provided dtype/layout assertions can be added in a later schema version; they must never override
contradictory graph facts.

For a full cache, `capacity >= max_sequence_length`, even if attention uses a local mask. For an evicting cache,
`capacity >= local_window_size`; the absolute sequence bound stays independent and can be much larger.
An explicit window assertion must match the graph. EP eligibility remains a separate check: for example, CUDA's
windowed GQA additionally requires `capacity == local_window_size` and the required in-place aliases.

The resolver requires declared pairs to match GQA input 3/output 1 or input 4/output 2, at graph inputs/outputs.
Every recognized pair must be declared, including a potentially nonempty output from an initially empty past.
It rejects multiple consumers instead of guessing which consumer determines location. It operates on one graph
scope/revision and retains no graph pointers; rewrites require re-resolution.
This is **not** a generic detector of arbitrary recurrent state. Strict whole-model admission must also reject
stateful patterns without an adapter; passing this GQA resolver alone does not certify such graphs.
The helper does not recursively discover nested-graph state; integration must resolve supported scopes or reject them.

## What goes into S0?

The full declaration stays with ORT for M2/R3/I1/I2 and the owner handshake. EP estimators receive only the facts
on which their estimates depend, projected before publishing a fresh immutable query snapshot:

| Projection | S0 representation |
|:-----------|:------------------|
| Maximum batch and allocated input cache capacity | Each exact past input in `max_input_shapes`: `[max_batch_size, kv_heads, capacity, head_size]` |
| Logical length envelope | `workload_bounds`: `declared_state.batch_size` = `[1, max_batch_size]`, `declared_state.total_sequence_length` = `[0, max_sequence_length]` |
| Per-pair alias promise | `node_constraints[].input_output_aliases`: input/output indices plus `Identical` or `Disjoint` |

Window attributes are already in the graph. Owner identity, lifetime, instance count, templates, and location
policy do not enter the EP query. No KV allocation is added to an L1 implementation-workspace requirement.
The existing GQA estimator is not adapted here; consuming this projection is a C2/C3 follow-up.

An unlisted alias pair, or an explicit `Unknown`, permits no sharing assumption. Indices refer to the query's
source-node schema, not a later lowered/fused node. The resolver establishes those indices; the S0 validator
rejects duplicate pairs and invalid alias enum values. Other producers must also establish valid port indices.

Projection preserves existing EP pins, permitted policies, unrelated bounds, and shape overrides. Conflicting
capacity/known-alias/bound facts fail; an explicit `Unknown` alias can be refined by the resolved declaration.
Changes to capacity, alias, logical envelope, graph, or other estimate-relevant facts require a new query snapshot
and new estimates, even if diagnostic IDs are unchanged. Ownership/lifetime/live-instance-only changes require
ORT backing/admission re-evaluation, but do not by themselves change the EP projection. Never project into a
query that has already been published or retained by candidates.

The input-capacity shapes are explicit BNSH state bounds. They are not proof that arbitrary downstream shape
inference is monotonic or that every activation/workspace term is bounded.

## Memory and binding obligations for integration

For one resolved tensor, let `S = batch * kv_heads * capacity * head_size * sizeof(dtype)`.
The helper checks `S` for `size_t` overflow and reports it as a per-tensor bound, not an allocation or reservation.
M2 must validate physical backing before charging `S` once for an identical pair or up to `2*S` for a disjoint
pair. For K and V together with equal shapes, that is `2*S` or `4*S` per layer. Multiply per-generator state by
`max_live_instances`; do not multiply per-session state. Allocator overhead, copy scratch, implementation
workspace, initialization overlap, and other live tensors remain separate accounting terms.

Binding must validate actual graph-compatible dtype/shapes, the admitted capacity and workload envelope,
live-instance count, exact alias/disjointness, lifetime, and planned memory locations. A promise alone must not
deduplicate bytes without subsequent enforcement. Reallocation above the envelope requires a new admission;
falling back to unbudgeted staging or a different kernel is not an allowed repair.

## Transport and follow-ups

The proposed external transport is one JSON session-config entry, tentatively `session.declared_state`.
**This PR does not register or consume that key. Do not enable it in GenAI yet.** The integration must accommodate
the existing 8192-byte config-value limit for realistic layer groups and expose positive, versioned support negotiation.
Successful insertion of an arbitrary config entry, ORT version alone, or the memory-location APIs alone is
not feature detection: an older runtime may ignore the unrecognized entry.

K3 connects parsing/resolution to session creation, planning, commitment, and bind-time checks; M2/R3/I1 consume
the full resolved declaration, while C1/C2 consume the S0 projection. I3 chooses placements, not K1.
K4 adds GenAI per-layer allocation; K5 adds an ORT-managed alternative with the same handshake, after K5-eval.
Neither ownership decision blocks this contract draft. Without opting into the eventual supported path, the
existing GenAI and ORT behavior remains unchanged.
