# Playbook 01: Repo Map and Architecture Overview

## Outcome

By the end of this playbook, you will understand where major subsystems live, how inference flows through ONNX Runtime, and where to start reading code for common contribution types.

## Start Here

- [README.md](../../README.md)
- [CONTRIBUTING.md](../../CONTRIBUTING.md)
- [docs/Coding_Conventions_and_Standards.md](../Coding_Conventions_and_Standards.md)
- [AGENTS.md](../../AGENTS.md)

## Core Runtime Mental Model

ONNX Runtime inference pipeline:

1. Load model
2. Build graph
3. Optimize graph
4. Partition across execution providers
5. Execute kernels

Use this model as your map when navigating code.

## Top-Level Repo Areas

- [onnxruntime/core](../../onnxruntime/core): graph, optimizer, session, runtime framework, execution providers
- [onnxruntime/test](../../onnxruntime/test): C++ tests, provider tests, model tests
- [onnxruntime/contrib_ops](../../onnxruntime/contrib_ops): non-standard operators and registrations
- [orttraining](../../orttraining): training-specific runtime code
- [include](../../include): public headers, including C API
- [docs](../): contributor and design references
- [csharp](../../csharp), [java](../../java), [js](../../js), [objectivec](../../objectivec), [rust](../../rust): language bindings

## Core Layers and Entry Points

Session and execution:

- [onnxruntime/core/session/inference_session.h](../../onnxruntime/core/session/inference_session.h)
- [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc)

Graph model and traversal:

- [onnxruntime/core/graph](../../onnxruntime/core/graph)

Optimization framework:

- [onnxruntime/core/optimizer/graph_transformer.cc](../../onnxruntime/core/optimizer/graph_transformer.cc)
- [onnxruntime/core/optimizer/rule_based_graph_transformer.cc](../../onnxruntime/core/optimizer/rule_based_graph_transformer.cc)

Execution providers:

- [onnxruntime/core/providers](../../onnxruntime/core/providers)

Provider and kernel tests:

- [onnxruntime/test/providers](../../onnxruntime/test/providers)

ONNX model test runner:

- [onnxruntime/test/onnx/main.cc](../../onnxruntime/test/onnx/main.cc)

## Contribution Path to Code Area

- If you are fixing model load or run behavior: start in session and graph folders.
- If you are adding graph rewrites: start in optimizer and related tests.
- If you are adding kernel support: start in providers and provider tests.
- If you are exposing new public APIs: start in include and C API docs.

## 45-Minute Code Reading Exercise

1. Open [onnxruntime/core/session/inference_session.h](../../onnxruntime/core/session/inference_session.h) and locate load, initialize, and run declarations.
2. Open [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc) and follow the high-level call flow.
3. Open one optimizer pass such as [onnxruntime/core/optimizer/conv_activation_fusion.cc](../../onnxruntime/core/optimizer/conv_activation_fusion.cc) to understand rewrite style.
4. Open a provider test directory such as [onnxruntime/test/providers/cpu](../../onnxruntime/test/providers/cpu) and inspect test naming and utilities.

## Common Failure Modes

- Reading files without a pipeline map: you lose context quickly.
- Starting in provider code before understanding session lifecycle.
- Attempting broad changes before finding existing tests in the same area.

## Exit Checklist

- [ ] You can describe the load to execute pipeline in your own words.
- [ ] You know where graph, optimizer, session, provider, and tests live.
- [ ] You identified one likely code area for your first code contribution.