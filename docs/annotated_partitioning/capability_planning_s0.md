# S0: Internal Capability Planning Protocol

This draft implements the internal information model for **S0.1-S0.4** of the
[region-aware placement proposal](https://github.com/microsoft/onnxruntime/pull/33372).
Its allocation vocabulary also supports the
[policy-based workspace estimation proposal](https://github.com/microsoft/onnxruntime/pull/32944).

The new types and consistency checks are not wired into discovery or execution.
`GetCapability`, existing optimization hooks, resource-accountant behavior, and
published C callbacks/API tables remain unchanged. There is no CPU/GPU selector,
budget admission, new kernel policy implementation, or KV-cache manager in S0.

| Item | Internal representation | Responsibility |
|:-----|:------------------------|:---------------|
| S0.1 | `CapabilityPlanningQuery` | An immutable, owned snapshot of the source graph identity, workload, normalized constraints, execution device/configuration, and execution-scope assumptions. |
| S0.2 | Optional `ComputeCapability::planning` containing `CapabilityPlanningInfo` | Source coverage stays in `sub_graph`; the payload describes implementation recipes, views/compatibility, contracts, requirements, and unresolved terms. |
| S0.3 | `CapabilityPlanningSelection` and immutable `CapabilityPlanningBinding` | Retain the selected identity, recipe, contract state, source membership, and originating query through subsequent stages. |
| S0.4 | `CapabilityPlanningSupport`, `CapabilityPlanningQueryResult`, and the C mapping below | Distinguish planning-safe participation and failure from empty discovery; specify a compatible opaque-C transport without publishing it yet. |

## S0.1: Query Snapshot

ORT publishes a query using `std::make_shared<const CapabilityPlanningQuery>`.
EPs and bindings share that snapshot. Do not create mutable aliases to published
queries, offers, or EP state. Graph/node views supplied alongside a query remain
borrowed; none of the retained common types stores a graph or node pointer.

`query_id` is a nonzero host-assigned diagnostic/transport identifier.
`graph_id` distinguishes graph scopes, including nested graphs; `graph_revision`
is scoped to that graph and may start at zero. Source-node indices and graph-value
names are meaningful only within this scope and revision. `ep_instance_id`
distinguishes provider instances, not just provider types.

The **owned query object is the in-process validity token**. An offer must retain
the same snapshot that the host supplied. A copied query with the same IDs is not
interchangeable. This avoids treating incomplete IDs or fingerprints as proof of
equivalence. The C adapter will attach the host's original snapshot to replies;
plugins will not manufacture C++ query objects.

Changes to graph revision, workload, configuration, constraints, tentative
ownership, device, or execution assumptions require a new snapshot. S0 has no
cache-retagging or descriptor-reuse mechanism. A later verified reuse mechanism
must prove that retained recipes, coverage, and bounds still apply.

### Bounds and Normalized Constraints

`MaxShapeOverrideMap` reuses existing input-shape override types. It does not turn
shadow-graph shape inference into an analytical upper-bound proof: downstream
shape transformations can be non-monotonic.

`workload_bounds` describes named nonnegative scalar ranges, including logical
sequence lengths and allocated cache capacity where relevant. An absent upper
bound is unknown. A known zero remains a valid bound. `max_concurrent_runs` and
`max_compute_streams` are optional positive bounds, not assertions that all
providers already support those modes.

`node_constraints` is the result of normalization, not another policy parser.
It has one entry per constrained source node. Required EP assignments are hard
pins. A missing `permitted_policies` preserves the configured policy; a present
list permits exactly its intents, and a present empty list permits none.
`KernelDispatchPolicy` reuses the hardware-neutral intent enum. `Auto` does not
grant permission to widen EP-private runtime routes.

`tentative_assignments` describes trial ownership for placement-sensitive
discovery. It is not committed placement. `ep_configuration` holds copied,
canonical effective configuration facts, including relevant defaults/build
features; it is not a retained session-options pointer or a channel for mutating
per-query provider options. The EP still interprets its private configuration.
Missing facts needed by an estimator must become unresolved terms.

Normalization, discovery dispatch, and actual support of execution modes belong
to C1. The initial integrated planner will use bounded workloads, serialized
Runs, and its validated single-stream scope.

## S0.2: Offers and Requirements

An execution candidate is atomic: select all of its source work or reject it.
Splitting it requires another offer/re-query, not slicing a fused estimate.
`IndexedSubGraph` remains the source-membership carrier. Its existing metadata or
the source graph supplies/derives boundary information.

Each candidate has one reproducible `recipe_id` and `implementation_id`.
Different recipes are different offers. Standalone kernels, registered fusion,
and compiled lowering share the same model. `prepared_state` retains an EP's
prepared match/layout/lowering metadata; it must not construct a device kernel
or allocate bulk device weights during discovery.

`alternative_view_ids` identifies the compatible sets in which an offer can
participate. Required/excluded candidate references are EP-instance-qualified.
Boundary constraints state layout/device agreements. Disjoint source coverage
alone is not a compatibility proof. Global view composition, companions,
interface validation, and complete coverage belong to F1/I1/F2.

`graph_effects` distinguishes an unchanged graph, a compact description, and
unresolved effects. Descriptions can specify boundaries, introduced tensors,
removed materializations, and aliases. Introduced tensor shapes/types can remain
unresolved for reporting, but must be resolved or conservatively bounded by the
whole-plan evaluator before admission. This compact schema is not an arbitrary
graph-transformation language; insufficient descriptions remain unresolved.

### Contract Identity and Coverage

A `contract_id` names retained EP-owned `resolved_state`, not an intent string
to resolve again during initialization or Run. Contracts separately describe
eligibility, admitted workload coverage, explicit failure conditions, and
allocation-bound status. An implementation can be executable but analytically
unresolved. A bounded contract can explicitly fail for requests with no permitted
bounded route; it need not promise successful computation for every request.

Multiple contracts are alternatives, never additional simultaneous charges.
S0 represents menus for fixtures; C3 will initially expose only one configured
GQA contract. I4 adds permitted alternative discovery/selection.

### Allocation Vocabulary

`allocation_requirements` describes **additional implementation-owned logical
allocations**, not a whole-session footprint or graph-inclusive resource total.
An unset list is unknown. An explicitly empty list plus `Bounded` asserts that
all relevant implementation-owned allocations are known absent.

Each allocation describes an identity, category, location, byte upper bound,
alignment, and lifetime. Optional size/location/alignment fields preserve
unknowns; a bounded claim cannot omit them. Known zero bytes and zero alignment
are different: the latter requests the allocator default, as in the existing
`WorkspaceRequirement`. Neither value proves zero physical allocator overhead.

| Existing L1 category | Planning allocation category |
|:---------------------|:-----------------------------|
| `runtime_workspace_bytes` | `ExecutionWorkspace` |
| `persistent_prepack_bytes` | `PersistentPrepack` |
| `initialization_scratch_bytes` | `InitializationScratch` |
| `runtime_transient_bytes` | `RuntimePreparation` |

C2 performs this adaptation using implementation sizing helpers. Legacy default
zeros do not establish proven zeros. Lifetimes must reflect the implementation,
including lazy preparation and device completion. `Operation` ends after the
operation's device work completes, not merely after its host launch returns.

Resource IDs are logical, not allocator addresses. Implementation allocations
are scoped by EP instance and candidate; graph-value references are scoped by
the query graph. Alias offsets must be explicit. `alias_of` and
`non_overlap_with` state enforceable relationships to validate during backing
planning, not permission to deduplicate by name or take an unverified maximum.
M1/M2 must establish location, extent, lifetime, and completion compatibility
before shared backing/reuse. No relationship is inferred from category tags.

ORT still accounts for graph tensors, external KV, transfers, allocator pools,
alignment/overhead, and simultaneous backing. `Other` accommodates explicit
implementation-owned storage not in the four L1 categories without charging
those framework-owned allocations again.

## S0.3: Selection, Retention, and Invalidation

`CreateCapabilityPlanningBinding` checks source/offer identity, the selected
candidate/recipe/view/contract IDs, a nonzero plan generation, eligibility,
resolved state/coverage, and complete bounded implementation requirements.
It also enforces normalized EP pins and permitted intents for covered nodes.
Unresolved offers remain reportable but cannot produce this bounded binding.
Failure returns a non-OK `Status` and leaves an existing output binding unchanged.

Validation returns statuses for caller-owned diagnostics. It does not initialize
the runtime environment or report expected planning rejections as Run failures.

The resulting const object copies selected metadata and source membership while
sharing the originating query, prepared recipe state, and selected resolved
contract state. It does **not** retain the whole contract menu. Discarding the
offer can therefore release unused contract state while selected state remains
valid. Copies of a binding share immutable state; they do not re-resolve policy.

`CapabilityPlanningState` provides an owned, opaque polymorphic host-state holder.
Only its EP knows the payload type. The last owner releases it through its
destructor. No framework code inspects route masks or invokes hidden graph
application methods through this holder.

`ValidateCapabilityPlanningBindingIdentity` guards the originating query and
active plan generation. It is an identity check, not a replacement for later
stage-specific reconciliation:

| Stage | Required follow-on behavior |
|:------|:----------------------------|
| Recipe application | F2 applies the selected prepared recipe only, with validated source provenance. Reconcile actual effects before commitment. |
| Initialization | C3/I2 propagate the same binding into registered-kernel construction or compilation; no independent contract resolution. |
| L2 verification | Check actual implementation requirements against the selected bounds and backing assumptions before allocation/use. |
| Run/dispatch | C3 consumes the retained permissions and enforces coverage/failure conditions. Compare the active plan generation. |

The source graph revision remains the planning origin; applying a verified
recipe is not permission to overwrite that origin with a new graph revision.
F2/I2 must track verified replacement provenance in the final plan. Changed
interfaces, larger bounds, changed lifetimes, or uncovered runtime states require
verified re-planning before incompatible allocation/use or explicit failure.

Rejected trials and initialization failures release their owned descriptors and
bindings. Committed session/kernel owners keep selected state alive until last
use, including outstanding device work. C3 must arrange that ownership; S0 does
not install session/kernel owners yet.

## S0.4: C Transport and Failure Semantics

The published callback remains:

```cpp
ORT_API2_STATUS(GetCapability, _In_ OrtEp* this_ptr, _In_ const OrtGraph* graph,
                _Inout_ OrtEpGraphSupportInfo* graph_support_info);
```

It need not be overloaded or replaced. A future versioned extension can use its
already-opaque support carrier:

1. The host checks explicit planning-safe feature support before making a
   Planning query. API version alone is not an opt-in or a side-effect guarantee.
2. The host owns the immutable query and puts a borrowed read-only C view in
   `OrtEpGraphSupportInfo`. Versioned `OrtEpApi` accessors expose its facts.
3. New C builders describe planning offers/contracts/resources. The adapter
   constructs optional internal metadata and attaches the original host snapshot.
4. An opaque EP state handle has documented ownership transfer **on success**
   and a release callback. On builder failure the EP retains ownership. An
   internal wrapper keeps the EP/factory dependencies alive and invokes release
   exactly once when its last selected/offer owner is gone.
5. Versioned selected-binding accessors must reach both compiled-region and
   registered-kernel creation, and retain state beyond call-scoped graph views.

These are mapping requirements, **not newly published API names or layouts**.
No STL objects, C++ virtual classes, borrowed graph pointers, or `shared_ptr`
representations cross the C ABI. New public table members would be append-only
with version checks before access, following repository API-versioning rules.
The current adapter is not converted in S0.

In-tree shared providers use opaque C++ `ComputeCapability` wrappers rather than
the host object's field layout. C1 will need corresponding private-bridge access
to query/metadata; S0 does not add bridge methods or access host fields through
those wrappers. Internal C++ consumers are rebuilt together as usual.

Legacy providers retain ordinary discovery. Strict planning needs explicit
planning-safe metadata, a proven adapter, or rejection. An older provider's empty
result cannot be interpreted as either planning support or known-zero memory.
Planning must bypass additive acceptance/resource commitment; a null accountant
is not a substitute for requirement discovery.

The future dispatcher returns `Status` separately from
`CapabilityPlanningQueryResult`; this draft does not add its signature. See
[Capability Planning Interfaces](capability_planning_interfaces.md) for the
proposed planning virtual (`GetExecutionCandidates`) and dispatcher.
`ValidateCapabilityPlanningResult` accepts completed, planning-safe empty
discovery and reportable unresolved contracts, but rejects non-participation,
missing metadata, stale snapshots, and malformed identities/bounded claims.

| Outcome | Meaning |
|:--------|:--------|
| Non-OK callback/dispatcher `Status` | Actual discovery/provider error; preserve it, never turn it into successful empty discovery. |
| `Unsupported` | No planning-safe participation; strict use reports `NOT_IMPLEMENTED`. |
| `PlanningSafe`, no offers, OK `Status` | Discovery completed; no eligible offers for this query. |
| Offer with `Unresolved` requirements | Keep diagnostics for reporting; not known zero or selectable by the bounded-binding helper. |
| Stale/malformed selection or violated pin | `INVALID_ARGUMENT`; do not install a binding. |
| Bounded local binding | Selected EP requirements are retained; **not** whole-session admission or a budget guarantee. |

## Worked Protocol Traces

These traces exercise the schema, not two new kernel/fusion implementations.
The host-only framework fixtures do not execute CUDA or prove estimator coverage.

**GQA, unchanged graph.** For the proposal's FP16 example (SM80/108 SMs,
8 query heads/2 KV heads, head size 512, batch at most 1, query length at most 4,
total KV/capacity at most 64), a permitted menu contains `C_memory` with
213,248-byte execution workspace and `C_latency` with 1,245,440 bytes.
The three other L1 categories are known zero for this particular configuration,
not by default. Memory state retains Flash plus its covering unfused fallback;
it is not a Flash-only promise. Selecting one contract retains that exact state
and all four categories; the other contract is not charged or retained by the
binding. Actual GQA policy discovery/dispatch wiring belongs to C3/I4.

**Illustrative fusion.** A source capability covers `A,B` for `A -> T -> B`,
belongs to an optimized view, and retains prepared recipe `R_AB`. It describes
eliminating internal materialized `T`, introducing initializer `P`, preserving
the external interface, and separate device prepack/init/workspace requirements.
The introduced graph initializer and device prepack are distinct allocations
unless an explicit validated relationship establishes shared backing.
An external consumer of `T` would require preserving/remapping it or rejecting
this recipe. Binding does not run `optimization_func`; F2 later applies and
reconciles the selected recipe before commitment.
