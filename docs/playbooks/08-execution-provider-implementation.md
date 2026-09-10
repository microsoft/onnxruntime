# Playbook 08: Execution Provider Implementation

## Outcome

By the end of this playbook, you will be able to start an execution provider implementation plan, choose between in-tree and plugin paths, and complete a minimal first-op bring-up with focused validation.

This playbook uses [docs/ExecutionProvider_Playbook.md](../ExecutionProvider_Playbook.md) as the primary deep-dive source and adapts it into the contributor learning sequence.

## Start Here

- [docs/ExecutionProvider_Playbook.md](../ExecutionProvider_Playbook.md)
- [onnxruntime/core/providers](../../onnxruntime/core/providers)
- [onnxruntime/test/autoep](../../onnxruntime/test/autoep)
- [docs/cuda_plugin_ep/QUICK_START.md](../cuda_plugin_ep/QUICK_START.md)

## First Decision: In-Tree or Plugin

Follow this rule early:

- choose in-tree if the provider is intended to ship and evolve inside ORT source builds
- choose plugin if the provider needs independent versioning and runtime registration

Reference path for plugin scaffolding:

- [onnxruntime/test/autoep/library/example_plugin_ep](../../onnxruntime/test/autoep/library/example_plugin_ep)

Do not start coding until this choice is explicit because it affects build wiring, API surface, and tests.

## EP Responsibilities Checklist

Any EP path should eventually cover:

1. device discovery and capability filtering
2. graph partitioning via `GetCapability`
3. execution path via kernel registry or compile callbacks
4. allocator and memory info handling
5. data transfer and optional stream synchronization
6. provider options and session integration
7. registration, correctness, and regression tests

Treat missing items as planned milestones, not optional details.

## Suggested Milestones for New Contributors

### Milestone 1: Skeleton + Build Wiring

- compile a minimal EP skeleton
- wire factory and options scaffolding
- confirm session can construct or register the EP

### Milestone 2: Capability + One Op

- implement conservative `GetCapability`
- claim one operator pattern only
- avoid broad claims until execution path is proven

### Milestone 3: Execution Path

- kernel registry path: register one working kernel
- compile path: return one valid compiled partition path
- confirm fallback behavior for unsupported nodes

### Milestone 4: Tests + Hardening

- add positive and negative partitioning tests
- add one execution correctness test against CPU baseline
- validate error handling for invalid options and unsupported cases

Use the full EP playbook for extended milestones after these core steps.

## Practical Path A: In-Tree EP

For in-tree EP work, use [docs/ExecutionProvider_Playbook.md](../ExecutionProvider_Playbook.md) sections on Path A as the implementation guide and keep this contributor flow:

1. start with provider skeleton and factory creator
2. wire CMake for provider build inclusion
3. add minimal capability logic
4. add first executable kernel or compile callback
5. add tests under provider test directories

If your first PR is too broad, split by milestone.

## Practical Path B: Plugin EP

For plugin EP work, start from the example plugin implementation:

- [onnxruntime/test/autoep/library/example_plugin_ep/example_plugin_ep.cc](../../onnxruntime/test/autoep/library/example_plugin_ep/example_plugin_ep.cc)
- [onnxruntime/test/autoep/library/example_plugin_ep/ep_factory.cc](../../onnxruntime/test/autoep/library/example_plugin_ep/ep_factory.cc)
- [onnxruntime/test/autoep/library/example_plugin_ep/ep.cc](../../onnxruntime/test/autoep/library/example_plugin_ep/ep.cc)

Then validate registration and device selection flow with:

- [onnxruntime/test/autoep/test_registration.cc](../../onnxruntime/test/autoep/test_registration.cc)
- [onnxruntime/test/autoep/test_selection.cc](../../onnxruntime/test/autoep/test_selection.cc)

Use [docs/cuda_plugin_ep/QUICK_START.md](../cuda_plugin_ep/QUICK_START.md) for runtime registration mechanics.

## First-Op Bring-Up Rules

- support one op, one dtype, one simple shape pattern first
- validate against CPU EP outputs before adding features
- keep `GetCapability` deterministic and conservative
- do not expose allocator or stream features until behavior is correct

This minimizes debugging surface and makes regressions easier to isolate.

## Test Strategy

Use focused tests for the stage you are implementing:

- registration and factory tests
- capability and partitioning tests
- execution correctness tests
- fallback tests for unsupported patterns

For plugin EPs, start from existing autoep test patterns in [onnxruntime/test/autoep](../../onnxruntime/test/autoep).

## Common Failure Modes

- over-claiming nodes in `GetCapability` before execution path is complete
- forgetting to add provider kernel table wiring after implementing kernels
- exposing allocator infos that allocator creation code cannot satisfy
- missing lifecycle pairing for created runtime resources
- bundling build wiring, partitioning logic, and execution behavior into one unreviewable PR

## Exit Checklist

- [ ] EP path choice (in-tree or plugin) is explicit and documented.
- [ ] Minimal skeleton builds and registers successfully.
- [ ] One-op capability and execution path works end-to-end.
- [ ] Focused tests cover registration, partitioning, and correctness.
- [ ] Unsupported cases fall back cleanly with actionable errors.