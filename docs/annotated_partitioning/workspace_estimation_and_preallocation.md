# Workspace Estimation and Preallocation

## Status

- **Status:** Proposed
- **Tracking issue:** [#29775](https://github.com/microsoft/onnxruntime/issues/29775)
- **Related roadmap:** [Future Directions: Constrained Environment Partitioning](future_directions_constrained_env.md)
- **Current user documentation:** [Partitioning with Annotations and Memory Constraints](PartitioningWithAnnotationsAndMemoryConstraints.md)
- **CUDA workspace inventory:** [CUDA Kernel Workspace Inventory](cuda_kernel_workspace_inventory.md)

This document records the proposed policy for combining profile-guided workspace
measurements, pre-partition workspace estimation, and post-assignment workspace
declarations. It is a design proposal, not documentation of fully implemented behavior.

## Motivation

ONNX Runtime needs workspace information at two different times:

1. During graph partitioning, before kernel instances exist, to decide whether a node fits
   within an execution provider's memory budget.
2. During session finalization, after kernels and algorithm choices are available, to
   preallocate workspace and avoid runtime arena allocations.

ORT has two potential sources for this information:

- **Profiling:** measure allocations from representative inference runs.
- **Estimation/declaration:** calculate workspace requirements from node metadata, shapes,
  device properties, and kernel configuration.

Neither source is sufficient on its own:

- A profile covers only the shapes, control-flow paths, device, provider settings, and
  algorithms exercised during profiling.
- A pre-partition estimate may need to be conservative because final kernel and algorithm
  choices have not been made.
- A post-assignment declaration is too late to make the original partitioning decision.

The framework therefore needs one policy that combines these sources without double
counting memory.

## Goals

- Provide a conservative per-node workspace value during partitioning.
- Use precise workspace declarations for static preallocation when available.
- Reuse workspace across nodes according to execution-plan liveness.
- Retain safe fallback behavior for kernels without workspace support.
- Detect stale profiles and incomplete workspace declarations.
- Report the source and confidence of workspace values to users.

## Non-goals

- Guarantee exact memory use for arbitrary unbounded dynamic shapes.
- Replace execution-provider-specific kernel or algorithm selection.
- Infer safe bounds for data-dependent output shapes without an explicit contract.
- Solve layer-boundary-aware partitioning or dynamic weight swapping.

## Terminology

### Profiled workspace

Workspace observed during one or more inference runs. Profiled values are empirical and
must not be treated as a universal upper bound unless the deployment constrains inputs,
control-flow paths, provider settings, and hardware to the profiled configuration.

The current `NodeStatsRecorder` CSV field `total_temp_allocations` records cumulative bytes
requested through the accounting allocator during a node's execution. It does not record
allocation/free intervals, peak simultaneously-live bytes, alignment, or workspace slots.

### Level-1 workspace estimate

A kernel-independent function callable during partitioning, before a kernel instance
exists. It uses the node, inferred planning shapes, execution-provider state, and device
properties to produce a conservative workspace size.

Level 1 is used for memory-budget decisions in `GetCapability`.

### Level-2 workspace declaration

A declaration made after kernel creation and algorithm selection. It describes the
workspace slots required to execute the kernel, including size and properties needed by
the memory planner.

Level 2 is used for actual preallocation and post-assignment verification.

### Fallback workspace

The safety margin used when neither a compatible profile nor a Level-1 estimate is
available. The existing size-based resource-accounting path uses a 1.5x node cost:

```text
base memory = initializer bytes + output tensor bytes
fallback workspace = base memory * 0.5
total node cost = base memory + fallback workspace
```

The fallback is not an independent measurement and must be replaced, not added, when a
better workspace source is available.

## Proposed Decision

### Partitioning-time workspace resolution

For each candidate node, ORT resolves one workspace value:

| Available information | Workspace used for partitioning |
|---|---|
| Compatible Level-1 estimate only | Level-1 estimate |
| Compatible profile only | Profiled value plus the configured profile margin |
| Compatible profile and Level-1 estimate | Maximum of the two values |
| Neither | Existing fallback workspace |

Profile and estimation values are **not added together**. They describe the same resource.

The partitioning cost is:

```text
node cost = initializer memory + output memory + resolved workspace
```

The 1.5x fallback applies per node only when no compatible profile or Level-1 estimate is
available for that node. Once all relevant kernels provide conservative estimates, the
fallback is unnecessary for those kernels.

### Why use the maximum when both sources exist?

- A profile is an observed value and may be lower than a future workload.
- A Level-1 estimate may be conservative or may miss an implementation detail.
- Taking the maximum avoids double counting while preventing either source from reducing
  the other source's safety.
- Large disagreement is useful diagnostic information and should be logged.

An execution provider may classify a Level-1 estimate as exact for a specific configuration.
ORT should still compare it with a compatible profile during rollout to detect estimator or
instrumentation defects.

## Profile Compatibility

A profile must be rejected or downgraded when its execution context does not match the
current session. A profile key should include, at minimum:

```text
model graph and weight hash
node identity
execution provider
kernel/implementation version
device architecture and relevant device properties
input-shape signature
relevant execution-provider options
```

Additional keys may be required for libraries whose algorithm or workspace choices vary by
version, such as CUDA, cuDNN, or cuBLAS.

A node name alone is not a sufficient compatibility key.

### Profile margin

Profiled allocation is evidence of a lower bound, not proof of an upper bound. A deployment
may configure a margin for profile-only nodes. The margin should be applied only to the
workspace value, not to weights and outputs that are already accounted separately.

The default margin and whether a profile can be marked authoritative remain open decisions.

## Dynamic Shapes

A maximum input-shape override produces a planning shape, not the actual runtime shape.
It is useful only when:

- runtime inputs remain within the configured bounds;
- shape inference propagates a valid upper bound; and
- output sizes are shape-dependent rather than data-dependent.

Operators with data-dependent output sizes, unexecuted control-flow branches, or incomplete
shape inference may remain unresolved.

For preallocation intended to prevent out-of-memory failures, ORT should eventually choose
one of these policies:

1. Enforce configured maximum input shapes at runtime.
2. Fall back to dynamic allocation when a runtime shape exceeds the plan.
3. Fail with a clear error in a strict constrained-memory mode.

Silently using a preallocation plan for inputs larger than its planning shapes is unsafe.

## Level-2 Verification and Preallocation

After kernel creation, each instrumented kernel declares one or more workspace slots. A
workspace requirement should eventually include:

```text
slot identifier
size in bytes
alignment
device and memory type
stream or concurrency constraints
```

The Level-2 declaration is the preferred description for actual allocation because it is
made after the kernel and algorithm are known.

When a compatible profile is also available:

```text
planned workspace = max(Level-2 declared peak, profiled peak)
```

During migration, if the profile is larger than the Level-2 declaration, ORT should treat
the difference as evidence of hidden or incomplete workspace declarations. Possible
responses are:

- add an opaque residual workspace slot;
- retain arena fallback for the undeclared portion; or
- fail initialization in a strict validation mode.

Level 2 should also be checked against the Level-1 value used during partitioning. If the
Level-2 requirement exceeds the partitioning reservation, ORT must not silently proceed as
if the original budget decision were still valid.

Potential responses include failing initialization, repartitioning, or using a documented
runtime fallback. The initial implementation should prefer an explicit error over hidden
budget violations.

## Workspace Memory Planning

Workspace requirements should participate in the execution memory plan as synthetic
allocations.

For sequential execution:

- workspace becomes live when a node begins execution;
- workspace becomes dead when that node finishes; and
- memory can be reused by later nodes.

The required allocation is therefore based on overlapping lifetimes, not the sum of every
node's workspace. With strictly sequential execution, this is often close to the largest
per-node workspace requirement.

For parallel or multi-stream execution, workspaces for nodes that may overlap must not share
the same memory region without an ordering dependency. The planner must account for graph
dependencies, stream assignment, and synchronization.

Workspace may be placed in the same reserved device buffer as activation memory if the
planner can represent both lifetimes and alignment requirements correctly. Otherwise, a
separate preallocated workspace buffer per device is an acceptable first implementation.

## Profiling Improvements Needed for Preallocation

The current `total_temp_allocations` field is sufficient as a conservative budgeting signal,
but it is not sufficient to produce an efficient workspace layout.

Preallocation-oriented profiling should capture:

```text
allocation size
allocation and free order
alignment
allocator/device
stream
peak simultaneously-live bytes
```

This trace can be reduced to a per-node peak and optional slot/lifetime information before
being persisted. Persisting raw pointer values is neither required nor desirable.

The current cumulative allocation value may exceed the true peak when a kernel allocates,
frees, and reallocates temporary buffers sequentially. Using it for preallocation is safe in
that case but may waste memory.

## Runtime Validation

ORT should optionally compare actual workspace allocations with the selected plan:

| Condition | Proposed behavior |
|---|---|
| Actual workspace is within the plan | Continue |
| Actual workspace exceeds the plan in permissive mode | Log and use arena fallback |
| Actual workspace exceeds the plan in strict mode | Fail with node, source, planned, and actual sizes |
| Profile and estimator differ substantially | Log source-specific diagnostics |

Validation should identify the node, kernel, EP, device, planning shape, selected workspace
source, and all candidate values.

## Reporting

User-visible resource reporting should separate:

```text
initializer/weight memory
activation/output memory
workspace memory
total planned device memory
workspace source coverage
```

Workspace reporting should include how many nodes used:

- Level-1 estimation;
- compatible profiling;
- both sources;
- Level-2 declaration; and
- fallback estimation.

The final summary belongs at the framework partitioning/session boundary, not inside an
individual execution provider's `GetCapability` implementation.

## Rollout

### Phase 1: Partitioning integration

- Define a framework-owned workspace estimate/result type with source metadata.
- Register Level-1 estimators for pilot kernels.
- Load compatible profile values through the same resolver.
- Replace the 1.5x fallback per node when a better source is available.
- Report source coverage and discrepancies.

### Phase 2: Level-2 declarations

- Add workspace slot declarations after kernel creation.
- Verify Level 2 against Level 1 and profiling.
- Retain current runtime allocation behavior.

### Phase 3: Static preallocation

- Extend the allocation planner with workspace lifetimes and alignment.
- Route declared workspace requests to planned buffers.
- Keep arena fallback for unresolved or dynamic requirements.
- Add strict validation for constrained deployments.

### Phase 4: Broader coverage

- Add estimators and declarations to high-workspace kernels.
- Generalize framework enforcement and reporting beyond CUDA.
- Remove fallback margins only where source coverage is complete.

## Open Decisions

1. What safety margin should apply to profile-only workspace?
2. Can a profile be marked authoritative, and under what deployment contract?
3. Should exceeding a maximum-shape override fail or fall back?
4. What happens when Level 2 exceeds Level 1 after partitioning?
5. Should the first preallocation implementation share the activation buffer or use a
   separate per-device workspace buffer?
6. How should parallel execution and multiple streams express workspace overlap?
7. What profile compatibility fields are mandatory for each EP?
8. How should fused-node workspace be derived from constituent nodes?

## Decision Log

| Date | Decision | Status |
|---|---|---|
| TBD | Profiling and estimation describe the same workspace and are not summed | Proposed |
| TBD | Use the maximum compatible value when profiling and Level 1 are both present | Proposed |
| TBD | Apply the 1.5x fallback only to nodes without a better workspace source | Proposed |
| TBD | Use Level-2 declarations for allocation and post-assignment verification | Proposed |
| TBD | Plan workspace according to execution overlap rather than summing all nodes | Proposed |

