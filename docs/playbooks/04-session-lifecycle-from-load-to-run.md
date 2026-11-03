# Playbook 04: Session Lifecycle from Load to Run

## Outcome

By the end of this playbook, you will be able to trace how ONNX Runtime turns a model into an executable session and identify where to debug or modify behavior in the `Load`, `Initialize`, and `Run` phases.

This playbook assumes you have already completed [Playbook 01](01-repo-map-and-architecture-overview.md) and [Playbook 02](02-build-test-and-debug-locally.md).

## Start Here

- [onnxruntime/core/session/inference_session.h](../../onnxruntime/core/session/inference_session.h)
- [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc)
- [onnxruntime/core/graph](../../onnxruntime/core/graph)
- [onnxruntime/core/optimizer](../../onnxruntime/core/optimizer)

## Mental Model

The session lifecycle is easiest to understand in three phases:

1. `Load`: construct or deserialize the in-memory model representation.
2. `Initialize`: register providers, transform and partition the graph, build session state, and finalize kernels.
3. `Run`: validate feeds and fetches, prepare execution state, and execute the graph.

If you are debugging a behavior change, locate the earliest phase where the state first becomes wrong.

## Phase 1: Load

Primary entry points:

- [onnxruntime/core/session/inference_session.h](../../onnxruntime/core/session/inference_session.h)
- [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc)

Key things that happen during `Load`:

- model format is selected as ONNX or ORT format
- model bytes, stream, or path are converted into an in-memory `Model`
- custom schemas and interop domains may be registered before the model is finalized
- strict shape/type inference and external initializer options are applied

What to inspect when debugging `Load`:

- whether the input is treated as ONNX or ORT format
- whether model parsing or protobuf loading fails
- whether external initializer settings or custom schema registration changes behavior

## Phase 2: Initialize

Primary entry point:

- [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc)

`Initialize()` is the main session-construction phase. Important checkpoints are:

1. Verify a model has been loaded.
2. Ensure a CPU execution provider exists, adding the default CPU EP if needed.
3. Create `SessionState`.
4. Register kernel registries from execution providers.
5. Add predefined graph transformers.
6. Run `TransformGraph()`.
7. Call `graph.Resolve()`.
8. Finalize session state with `FinalizeSessionState()`.
9. Resolve memory-pattern flags and mark the session initialized.

This is the phase where most contributor-facing runtime decisions are made.

## What `TransformGraph()` Does

The `TransformGraph()` flow in [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc) is especially important because it connects graph rewriting with execution provider assignment.

High-level order:

1. inline functions ahead of time
2. normalize QDQ structure when needed
3. apply level 0 and level 1 graph optimizations
4. partition the graph based on execution provider capabilities
5. run level 2 and level 3 optimizations, possibly in a loop
6. insert cast nodes
7. run level 4 optimizations
8. insert copy nodes

This order matters. If you change a graph transform or provider capability rule, verify which stage is supposed to own that behavior.

## Phase 3: Run

Primary entry points:

- [onnxruntime/core/session/inference_session.h](../../onnxruntime/core/session/inference_session.h)
- [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc)

The public `Run()` overloads funnel into `RunImpl()`.

Key things that happen during `RunImpl()`:

- optional per-run profiling is configured
- run-level graph capture settings are checked
- session initialization is validated
- inputs and requested outputs are validated
- feeds/fetches bookkeeping is created
- execution providers receive `OnRunStart()` callbacks
- the graph is executed through `utils::ExecuteGraph(...)`
- execution providers receive `OnRunEnd()` callbacks
- optional cleanup and memory-arena shrinkage are performed

If you are debugging inference-time failures, this is the best place to separate:

- input validation issues
- execution-provider startup/shutdown issues
- actual kernel execution failures
- stream cleanup or graph capture issues

## Code Reading Path

Follow this sequence when learning the pipeline:

1. Read the `Load`, `Initialize`, and `Run` declarations in [onnxruntime/core/session/inference_session.h](../../onnxruntime/core/session/inference_session.h).
2. Read the `Load` implementations in [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc).
3. Read `Initialize()` in [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc).
4. Read `TransformGraph()` in [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc).
5. Read `RunImpl()` in [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc).

Do not start by reading all call sites. The core behavior is controlled here.

## Practical Debugging Recipe

Use this sequence when a model behaves differently than expected:

1. Confirm whether the issue appears during model load, session initialization, or execution.
2. Add or run the smallest test that reproduces the issue.
3. If the graph looks wrong before execution, inspect `TransformGraph()` and `graph.Resolve()`.
4. If the graph is correct but execution is wrong, inspect provider callbacks and `utils::ExecuteGraph(...)`.
5. If behavior depends on provider assignment, verify partitioning and inserted copy nodes.

## Common Failure Modes

- Registering execution providers or graph transformers after `Initialize()` and expecting them to take effect.
- Debugging kernel output before checking whether graph partitioning already changed the graph.
- Treating a load-time parse failure like a run-time execution failure.
- Missing the difference between ORT format model loading and ONNX model loading.

## Exit Checklist

- [ ] You can explain what work belongs to `Load`, `Initialize`, and `Run`.
- [ ] You know that `TransformGraph()` is the main bridge between graph optimization and provider partitioning.
- [ ] You know where `SessionState` is created and finalized.
- [ ] You have a debugging strategy for locating issues in the correct session phase.