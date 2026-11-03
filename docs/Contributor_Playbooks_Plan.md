# Contributor Playbooks Plan

This document lays out a proposed set of 10 playbooks for new contributors to ONNX Runtime. The goal is to turn the repo’s scattered reference material into a guided learning path that starts with low-risk changes and builds toward deeper subsystems.

The current repo already has one true playbook: [ExecutionProvider_Playbook.md](ExecutionProvider_Playbook.md). The rest of the guidance is spread across reference docs such as [CONTRIBUTING.md](../CONTRIBUTING.md), [PR_Guidelines.md](PR_Guidelines.md), [Coding_Conventions_and_Standards.md](Coding_Conventions_and_Standards.md), [Model_Test.md](Model_Test.md), and [C_API_Guidelines.md](C_API_Guidelines.md).

## Intended Learning Order

The playbooks should be written in this sequence so that each one builds on the previous one:

1. Repo map and architecture overview
2. Build, test, and debug locally
3. First PR and environment setup
4. Session lifecycle from load to run
5. Adding or changing a kernel
6. Adding a contrib operator
7. Graph optimizations and fusion passes
8. Execution provider implementation
9. Memory management, allocators, and data movement
10. Python and C API binding extension

## Draft Status

The following playbooks have been drafted:

1. [01 Repo Map and Architecture Overview](playbooks/01-repo-map-and-architecture-overview.md)
2. [02 Build, Test, and Debug Locally](playbooks/02-build-test-and-debug-locally.md)
3. [03 First PR and Environment Setup](playbooks/03-first-pr-and-environment-setup.md)
4. [04 Session Lifecycle from Load to Run](playbooks/04-session-lifecycle-from-load-to-run.md)
5. [05 Adding or Changing a Kernel](playbooks/05-adding-or-changing-a-kernel.md)
6. [06 Adding a Contrib Operator](playbooks/06-adding-a-contrib-operator.md)
7. [07 Graph Optimizations and Fusion Passes](playbooks/07-graph-optimizations-and-fusion-passes.md)
8. [08 Execution Provider Implementation](playbooks/08-execution-provider-implementation.md)
9. [09 Memory Management, Allocators, and Data Movement](playbooks/09-memory-management-allocators-and-data-movement.md)
10. [10 Python and C API Binding Extension](playbooks/10-python-and-c-api-binding-extension.md)

## Proposed Playbooks

### 1. Repo map and architecture overview

Purpose: explain how the major pieces of ORT fit together before a contributor touches code.

Should cover:

- model loading, graph IR, optimization, partitioning, and execution
- top-level directories and what lives in each one
- where contributors should look for inference, training, and bindings work

Starter references:

- `onnxruntime/core/`
- `onnxruntime/test/`
- `orttraining/`
- `csharp/`, `java/`, `js/`, `objectivec/`, `rust/`

### 2. Build, test, and debug locally

Purpose: make the everyday developer loop predictable and reproducible.

Should cover:

- the recommended build entry points on each platform
- selecting targeted tests instead of full-suite runs
- common debugging workflows for kernel and graph issues
- how to use the existing test infrastructure effectively

Starter references:

- `build.sh`
- `build.bat`
- [Model_Test.md](Model_Test.md)

### 3. First PR and environment setup

Purpose: help a new contributor get from a fresh clone to a small, reviewable pull request.

Should cover:

- cloning the repo and setting up prerequisites
- choosing a good first issue or a safe small task
- running the smallest useful build and test loop
- preparing a PR that follows the repo’s conventions

Starter references:

- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [PR_Guidelines.md](PR_Guidelines.md)
- [Coding_Conventions_and_Standards.md](Coding_Conventions_and_Standards.md)

### 4. Session lifecycle from load to run

Purpose: show how ORT turns a model into an executable session.

Should cover:

- the `Load()` to `Initialize()` to `Run()` flow
- graph optimization and partitioning checkpoints
- how session options affect execution
- where execution providers and kernels enter the flow

Starter references:

- `onnxruntime/core/session/`
- `onnxruntime/core/graph/`
- `onnxruntime/core/optimizer/`

### 5. Adding or changing a kernel

Purpose: teach the smallest meaningful execution change that still touches ORT internals.

Should cover:

- how kernel registration works
- CPU vs provider-specific kernel structure
- shape/type validation and error reporting
- adding tests for correctness and edge cases

Starter references:

- `onnxruntime/core/framework/`
- `onnxruntime/core/providers/`
- `onnxruntime/test/providers/`

### 6. Adding a contrib operator

Purpose: show how to introduce a `com.microsoft` operator and keep it maintainable.

Should cover:

- schema definition and registration
- kernel implementations across providers as needed
- documentation and test coverage expectations
- compatibility concerns for a non-standard op

Starter references:

- [ContribOperators.md](ContribOperators.md)
- `onnxruntime/contrib_ops/`

### 7. Graph optimizations and fusion passes

Purpose: help a contributor understand how ORT rewrites graphs before execution.

Should cover:

- pattern matching and transformation basics
- when to fuse nodes and when not to
- how to keep optimizer behavior conservative and deterministic
- how to validate rewrites with focused tests

Starter references:

- `onnxruntime/core/optimizer/`
- `onnxruntime/core/graph/`

### 8. Execution provider implementation

Purpose: expand the existing playbook into a newcomer-friendly path from scaffold to first supported op.

Should cover:

- provider skeleton, factory wiring, and build integration
- capability discovery and partitioning
- allocators, memory info, and data movement
- runtime registration and tests

Starter references:

- [ExecutionProvider_Playbook.md](ExecutionProvider_Playbook.md)
- `onnxruntime/core/providers/`
- `docs/cuda_plugin_ep/QUICK_START.md`

### 9. Memory management, allocators, and data movement

Purpose: explain the performance- and correctness-critical memory layer.

Should cover:

- allocator types and lifetime rules
- device, pinned, and read-only memory concepts
- transfer paths across execution providers
- common failure modes and how to test them

Starter references:

- `onnxruntime/core/framework/`
- `onnxruntime/core/providers/`

### 10. Python and C API binding extension

Purpose: show how a C++ or session-level feature is exposed to Python and the public C API.

Should cover:

- where the public C API lives
- how Python bindings map onto the C API and core runtime
- compatibility and versioning constraints for exported APIs
- tests for binding and option propagation

Starter references:

- [C_API_Guidelines.md](C_API_Guidelines.md)
- `include/onnxruntime/core/session/onnxruntime_c_api.h`
- `onnxruntime/python/`

## Suggested Writing Rule

Each playbook should be practical and task-oriented:

- start with a short outcome statement
- list the repo files or subsystems the reader should open first
- walk through a minimal implementation path
- include a small test plan and common failure modes
- end with a checklist the contributor can use before opening a PR

## What Makes These 10 Useful

Together, the set covers the main contributor journeys in ORT:

- first-time contribution
- core runtime concepts
- local development and debugging
- operator and kernel work
- graph transformation work
- execution provider work
- memory and performance concerns
- public API and binding work

That gives new contributors a path from “I can build the repo” to “I can safely modify a core subsystem.”