# Capability Planning Interfaces

This draft defines planning-safe execution-provider (EP) discovery for the
[region-aware CPU/GPU placement proposal](https://github.com/microsoft/onnxruntime/pull/33372).
An **execution candidate** is a `ComputeCapability` with a `CapabilityPlanningInfo` payload: one implementation
of a source-node set, with alternative execution contracts. It is not a placement decision or a memory reservation.

**Decision:** keep legacy `GetCapability` unchanged; add `GetExecutionCandidates` with the same return type and
planning context as its fourth argument. Carry planning output in `ComputeCapability::planning`.

**Status:** [S0](capability_planning_s0.md) (#33841) supplies the payload types, validators, and binding helpers.
The discovery virtual, host dispatcher, and shared-provider bridge below are proposed C1 work, not implemented in S0.
The planner, whole-plan admission, recipe application, and runtime integration are outside this interface draft.

## Discovery API

```cpp
// IExecutionProvider, in include/onnxruntime/core/framework/execution_provider.h.
// Forward-declare outside the SHARED_PROVIDER include guard.
struct CapabilityPlanningQuery;

// Existing: unchanged.
virtual std::vector<std::unique_ptr<ComputeCapability>>
GetCapability(const onnxruntime::GraphViewer& graph_viewer,
              const IKernelLookup& kernel_lookup,
              const GraphOptimizerRegistry& graph_optimizer_registry,
              IResourceAccountant* resource_accountant = nullptr) const;

// Proposed: declare after the last existing virtual, under a distinct name.
virtual std::vector<std::unique_ptr<ComputeCapability>>
GetExecutionCandidates(const onnxruntime::GraphViewer& graph_viewer,
                       const IKernelLookup& kernel_lookup,
                       const GraphOptimizerRegistry& graph_optimizer_registry,
                       const std::shared_ptr<const CapabilityPlanningQuery>& query_context) const;
```

The first three parameters retain their roles. The host supplies the query's **source graph view**, before any
selected-placement filter. `query_context` is a validated, immutable snapshot containing:

| Query facts | Members |
|:------------|:--------|
| Identity | `mode = Planning`, `query_id`, `graph_id`, `graph_revision`, `source_view_id`, `ep_instance_id` |
| Workload | `max_input_shapes`, `workload_bounds` |
| Placement constraints | `node_constraints`, `tentative_assignments` |
| Execution assumptions | `execution_device`, `ep_configuration`, `max_concurrent_runs`, `max_compute_streams` |

Planning has no accountant: requirements are returned as data, not charged during discovery. Passing a null
accountant to legacy discovery is not equivalent; for example, CUDA skips its resource queries in that case.

An EP opts in by overriding the new virtual. The base default returns `{}` without calling legacy discovery.
Define it out of line in `execution_provider.cc`, and separately in `provider_bridge_provider.cc` for shared
providers; `ComputeCapability` is incomplete in `execution_provider.h`. The CPU EP also needs an explicit planning
override or adapter; its legacy kernel-lookup default does not establish planning support.

Discovery must be repeatable and read-only: no graph mutation, `optimization_func` invocation, compilation, kernel
creation, bulk device allocation, or resource commitment. Read-only optimizer selection functions are allowed.
The host calls each EP instance serially. Graph and node views are call-scoped and must not be retained.

## `CapabilityPlanningInfo`

**Definition:** `CapabilityPlanningInfo` is the immutable, query-specific description of **one candidate**. It
identifies an implementation, declares compatible graph views and companion constraints, describes how to realize
the implementation, and offers policy-specific execution contracts. Source-node membership stays in `sub_graph`.

The attachment in `ComputeCapability` is optional; existing members and constructors remain unchanged. Member-only
view:

```cpp
struct ComputeCapability {
  std::unique_ptr<IndexedSubGraph> sub_graph;
  std::function<Status(Graph&, const ComputeCapability&, ComputeCapability&, const GraphOptimizerRegistry&)>
      optimization_func;
  std::vector<std::unique_ptr<ComputeCapability>> nodes_to_optimize;
  std::shared_ptr<const CapabilityPlanningInfo> planning;
};
```

The complete S0 payload definition, from
[`capability_planning.h`](../../onnxruntime/core/framework/capability_planning.h):

```cpp
struct CapabilityPlanningInfo {
  std::shared_ptr<const CapabilityPlanningQuery> query;
  std::string candidate_id;
  std::string implementation_id;
  InlinedVector<std::string> alternative_view_ids;
  CapabilityCompatibilityConstraints compatibility;
  CapabilityImplementationRecipe recipe;
  InlinedVector<CapabilityExecutionContract, 1> contracts;
};
```

| Field | Meaning and invariant |
|:------|:----------------------|
| `query` | The exact snapshot passed to discovery, not a copy with matching IDs. It supplies graph revision, workload, configuration, and EP identity. |
| `candidate_id` | Non-empty and unique within this EP's result. Cross-candidate references also name the EP instance. |
| `implementation_id` | Non-empty identity of the implementation this candidate describes. |
| `alternative_view_ids` | One or more unique, non-empty view IDs in which the candidate can participate. |
| `compatibility` | Required/excluded candidates and boundary layout/device constraints. The planner enforces these relationships. |
| `recipe` | One identified realization: standalone kernel, registered fusion, compiled region, or explicitly unresolved. Different recipes are different candidates. |
| `contracts` | One or more contracts with distinct IDs. These are alternative policies for this implementation, not simultaneous memory charges. |

```mermaid
flowchart LR
  C["ComputeCapability"] --> N["sub_graph<br/>source-node membership"]
  C --> I["planning: CapabilityPlanningInfo"]
  I --> Q["query<br/>same immutable snapshot"]
  I --> V["alternative_view_ids<br/>compatibility constraints"]
  I --> R["recipe<br/>prepared state + graph effects"]
  I --> T["contracts[]<br/>policy + coverage + allocation bounds"]
```

### Recipe and execution contracts

A **recipe** says how to realize the candidate. A **contract** says under which policy and workload that
implementation is eligible, what it may allocate, and which EP-owned state must survive selection.

```cpp
using CapabilityPlanningStatePtr = std::shared_ptr<const CapabilityPlanningState>;

struct CapabilityImplementationRecipe {
  std::string recipe_id;
  CapabilityRecipeKind kind = CapabilityRecipeKind::Unresolved;
  CapabilityPlanningStatePtr prepared_state;
  CapabilityGraphEffects graph_effects;
  InlinedVector<std::string> unresolved_terms;
};

struct CapabilityExecutionContract {
  std::string contract_id;
  KernelDispatchPolicy policy = KernelDispatchPolicy::Auto;
  CapabilityContractEligibility eligibility = CapabilityContractEligibility::Unresolved;
  std::string eligibility_reason;
  std::string workload_coverage;
  InlinedVector<std::string> explicit_failure_conditions;
  CapabilityPlanningStatePtr resolved_state;

  CapabilityRequirementBoundStatus bound_status = CapabilityRequirementBoundStatus::Unresolved;
  std::optional<InlinedVector<CapabilityAllocationRequirement, 1>> allocation_requirements;
  InlinedVector<std::string> unresolved_terms;
};
```

| Supporting type | Content |
|:----------------|:--------|
| `CapabilityCompatibilityConstraints` | `required_candidates`, `excluded_candidates`, and `boundary_constraints`. |
| `CapabilityGraphEffects` | `Unresolved`, `Unchanged`, or `Described`; boundary inputs/outputs, introduced tensors, removed values, aliases, and unresolved terms. `Unchanged` cannot declare rewrites or alias changes. |
| `CapabilityAllocationRequirement` | Resource ID, category, device, byte upper bound, alignment, lifetime, and optional alias/non-overlap relationships. Categories include workspace, persistent prepack, initialization scratch, runtime preparation, and other implementation-owned storage. |
| `CapabilityPlanningState` | Opaque EP-owned host metadata. `prepared_state` retains match/lowering data; `resolved_state` retains the contract's resolved permissions. Neither is a materialized device kernel or borrowed graph view. |

**Unknown is not zero.** An unset `allocation_requirements` means unknown. A present empty list with `Bounded`
asserts no relevant implementation-owned allocations. A bounded list must identify every allocation's kind,
device, size, alignment, and lifetime, with resolved relationship identities and alias offsets where supplied.
Zero alignment requests the allocator default; it does not establish zero allocator overhead.

These are **additional implementation-owned allocations**, not a whole-session footprint. ORT separately accounts
for graph tensors, transfers, external KV storage, and allocator overhead. `Operation` lifetime ends after device
work completes, not after its host launch returns.

`ValidateCapabilityPlanningResult` checks non-empty, duplicate-free source membership, the original query pointer,
identities, views, and every contract. Unresolved recipes, effects, and requirements need diagnostics; non-eligible
contracts need an eligibility reason. Reportable unresolved contracts are valid discovery output, but not bindable.
Publish queries, payloads, and EP state as const without mutable aliases.

For example, the [S0 GQA fixture](capability_planning_s0.md#worked-protocol-traces) has candidate `GQA_42`,
implementation `GQA_cuda_fp16`, recipe `R_GQA`, and view `V_source`:

| Contract | Policy | Execution workspace |
|:---------|:-------|:--------------------|
| `C_memory` | `Memory` | 213,248 bytes |
| `C_latency` | `Latency` | 1,245,440 bytes |

Those are alternative bounds for that fixture's workload/configuration, not general GQA constants.

## Query and binding protocol

The planner calls a host dispatcher rather than the virtual directly. Proposed declarations:

```cpp
Status ValidateCapabilityPlanningQuery(const std::shared_ptr<const CapabilityPlanningQuery>& query);

Status QueryCapabilityPlanning(const IExecutionProvider& ep,
                               const GraphViewer& graph_viewer,
                               const IExecutionProvider::IKernelLookup& kernel_lookup,
                               const GraphOptimizerRegistry& graph_optimizer_registry,
                               const std::shared_ptr<const CapabilityPlanningQuery>& query_context,
                               CapabilityPlanningQueryResult& result);
```

Export S0's existing query validator and call it **before entering EP code**. The dispatcher header must include
`execution_provider.h` because `IKernelLookup` is nested in `IExecutionProvider`.

```mermaid
sequenceDiagram
  participant P as Planner
  participant D as QueryCapabilityPlanning
  participant E as EP GetExecutionCandidates
  participant B as CreateCapabilityPlanningBinding

  P->>D: query + source view + lookup + registry
  D->>D: reset result; validate query
  Note over D,E: EP is called only if the query is valid
  D->>E: GetExecutionCandidates(..., query_context)
  E-->>D: candidate vector
  D->>D: package and validate entire result
  D-->>P: Status + candidate set
  Note over P,B: Continue only with an OK result and a selected cover
  P->>B: query + candidate + selection
  B-->>P: Status + immutable binding
```

| Dispatcher result | Meaning |
|:------------------|:--------|
| OK, `PlanningSafe` | Validated, non-empty candidate set. Contracts may still be unresolved. |
| `NOT_IMPLEMENTED` | Empty return or missing `planning`: no planning-safe answer, **not known-zero cost**. |
| `INVALID_ARGUMENT` | Malformed query or candidate, including null entries, foreign query pointers, duplicate IDs, or incomplete bounded claims. |
| `FAIL` | The EP threw a `std::exception`; preserve its message. |

Keep all returned entries for validation; never silently drop invalid candidates. Any failure clears the result,
and diagnostics survive in the returned `Status`. One invalid candidate rejects the whole EP result.

The vector return cannot distinguish unsupported discovery from an opted-in EP finding no eligible candidates.
Both map to `Unsupported`. S0 itself supports `PlanningSafe` with an empty set, but this call shape cannot produce
that distinction. Exceptions are its only EP failure channel; without exception support, throwing aborts.

### Selecting and retaining a contract

```cpp
Status CreateCapabilityPlanningBinding(const std::shared_ptr<const CapabilityPlanningQuery>& query,
                                       const ComputeCapability& capability,
                                       const CapabilityPlanningSelection& selection,
                                       std::shared_ptr<const CapabilityPlanningBinding>& binding);
```

The selection names `candidate_id`, `recipe_id`, `alternative_view_id`, `contract_id`, and a nonzero
`plan_generation`. The binder validates the entire candidate, then requires a resolved recipe/effects and an
`Eligible`, `Bounded` contract with retained state and workload coverage. Compiled recipes or described graph
effects also require `prepared_state`. Covered nodes must satisfy the query's EP pins and permitted policies.

Malformed or ineligible selections return `INVALID_ARGUMENT`; unresolved recipes or selected bounds return
`NOT_IMPLEMENTED`. Failure leaves the existing binding unchanged. A malformed sibling contract blocks binding;
a valid unresolved sibling does not.

A binding copies source membership, compatibility, the recipe, and **only the selected contract**, sharing the
original query and selected states. Discarding the candidate set can release unselected states without losing the
selected ones. `ValidateCapabilityPlanningBindingIdentity` checks the active query and plan generation.
This is a local validity check, **not whole-plan feasibility or budget admission**.

Changes to graph, workload, placement, configuration, or execution assumptions require a new query snapshot.
Do not retag an old candidate or binding. Retained state and hooks must not capture borrowed graph/node pointers;
release candidates and bindings before their EP library unloads.

## Views and existing optimization hooks

Candidates may overlap during discovery; a selected cover must not. For source work `A -> B -> C`, an EP can offer
standalone candidates and a fused `A,B` alternative:

```mermaid
flowchart LR
  subgraph S["V_source cover"]
    A["cand_A: A"] -->|"T"| B["cand_B: B"] --> C["cand_C: C"]
  end
  subgraph O["V_opt cover"]
    AB["cand_AB: A,B<br/>RegisteredFusion"] --> CO["cand_C: C"]
  end
```

`cand_C` appears once in the result and lists both views. Selecting `cand_AB` together with `cand_B` is invalid
because `B` is covered twice. A fused/compiled candidate is atomic: accept it whole or reject it. A placement
boundary between `A` and `B` requires another cover or re-query. Eliminating `T` is valid only if external consumers
are preserved or remapped. Compatibility and whole-cover checks belong to the planner, not S0's local binder.

`nodes_to_optimize`, `optimization_func`, and optional `MetaDef` coexist with `planning`; their legacy meaning is
unchanged. Planning and binding never run application hooks. Later recipe application must apply the selected
implementation and reconcile its actual effects against `planning->recipe`. S0 does not check hook/recipe or
`MetaDef` consistency. Existing TensorRT and CUDA runtime-fusion producers (#33705, #33757) need explicit integration.

## EP boundaries and ABI

The recommended distinct-name virtual is **Alternative E**. Declare it after `GetEpContextDataCallbackRequirements`,
the last existing virtual, so existing slots stay in place on both Itanium and MSVC.

| Boundary | Required integration |
|:---------|:---------------------|
| In-tree EP | Override `GetExecutionCandidates` and attach `planning` directly. |
| Shared-provider C++ library | Use the opaque `ComputeCapability` wrapper; append `ComputeCapability__SetPlanning` to `ProviderHost`. Define the empty default locally, so it needs no host forwarding entry. |
| Plugin EP C ABI | Preserve the published `OrtEp::GetCapability`. Future version-gated C builders and explicit planning-safe opt-in map into internal metadata; no STL, C++ virtual objects, or `shared_ptr` representations cross this boundary. See S0.4. |

**Why a distinct name:** MSVC groups same-name virtual overloads at their first declaration, newest overload first.
Adding a `GetCapability` overload can therefore shift existing slots even when declared last. Itanium uses
declaration order. This MSVC analysis comes from Clang's ABI implementation, not an MSVC layout test.

`IExecutionProvider` does not have a strictly append-only history (`SupportsMoeOffload`, #33575); core and shared
providers are rebuilt together. Keeping slots stable is an additional safeguard, not permission to call the new
virtual on an older library. `ProviderLibrary::Load` has no version handshake. `ProviderHost` entries and published
C API tables retain their append-only constraints.

Letting provider builds include `capability_planning.h` extends existing C++ layout assumptions to its STL,
`InlinedVector`, `optional`, and shared-state members. Header compatibility under `SHARED_PROVIDER` remains unchecked.

| Alternative | Tradeoff |
|:------------|:---------|
| A. Same-name `GetCapability` overload | Keeps the proposal's name and existing overrides, but adds name hiding/overload mechanics and shifts MSVC slots. |
| B. One `GetCapability` keyed on `mode` | Removes overload mechanics, but changes every override and retypes the shipped `IExecutionProvider__GetCapability` bridge entry. |
| D. `Status GetExecutionCandidates(..., result)` | Explicit errors and support/empty-result semantics, but changes the return shape and exposes `CapabilityPlanningQueryResult` to EP code. |

## Remaining decisions

- **Empty results and failure channel:** keep the conservative vector convention, add an opt-in marker for
  known-empty discovery, or adopt Alternative D.
- **Shared providers:** confirm payload-header/layout compatibility and whether lockstep builds suffice or a library
  feature/version check is needed before calling the new virtual.
- **Integration:** choose the CPU planning adapter, minimal-build guard, and final API/header names.
- **Application:** decide how later stages reconcile stored hooks with prepared recipes.

The proposed discovery/dispatcher interfaces have not been compiled against ORT headers, the bridge has not been
checked, and the diagrams have not been rendered. Detailed payload semantics remain in [S0](capability_planning_s0.md).
