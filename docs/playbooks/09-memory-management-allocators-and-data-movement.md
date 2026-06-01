# Playbook 09: Memory Management, Allocators, and Data Movement

## Outcome

By the end of this playbook, you will be able to reason about allocator selection, tensor placement, and cross-device copies in ONNX Runtime, and debug common memory-transfer issues with focused tests.

This playbook assumes you have already completed [Playbook 04](04-session-lifecycle-from-load-to-run.md), [Playbook 05](05-adding-or-changing-a-kernel.md), and [Playbook 08](08-execution-provider-implementation.md).

## Start Here

- [include/onnxruntime/core/framework/allocator.h](../../include/onnxruntime/core/framework/allocator.h)
- [onnxruntime/core/framework/session_state.h](../../onnxruntime/core/framework/session_state.h)
- [onnxruntime/core/framework/data_transfer_manager.h](../../onnxruntime/core/framework/data_transfer_manager.h)
- [onnxruntime/core/framework/data_transfer_utils.h](../../onnxruntime/core/framework/data_transfer_utils.h)
- [onnxruntime/test/providers/memcpy_test.cc](../../onnxruntime/test/providers/memcpy_test.cc)
- [onnxruntime/test/providers/io_binding_test.cc](../../onnxruntime/test/providers/io_binding_test.cc)

## Mental Model

Memory behavior in ORT is the interaction of three layers:

1. allocator layer: where buffers are allocated and freed
2. session state layer: which allocators are available for each device and memory type
3. data transfer layer: how tensors move between devices and memory locations

If outputs are wrong, slow, or failing only on some providers, check which layer is responsible before changing code.

## Allocator Fundamentals

In [include/onnxruntime/core/framework/allocator.h](../../include/onnxruntime/core/framework/allocator.h), `IAllocator` defines allocation contracts, including stream-aware allocation support and memory-size safety helpers.

Important concepts to track:

- `OrtMemoryInfo` and `OrtDevice` identify allocator location and semantics
- arena behavior can be configured via `OrtArenaCfg`
- stream-aware allocators can use `AllocOnStream(...)`
- allocation size arithmetic should be overflow-safe

When implementing or modifying allocator usage, preserve these contracts first and optimize second.

## SessionState as the Allocation Router

In [onnxruntime/core/framework/session_state.h](../../onnxruntime/core/framework/session_state.h), `SessionState` owns allocator maps and exposes:

- `GetAllocator(const OrtMemoryInfo&)`
- `GetAllocator(const OrtDevice&)`
- `GetInitializerAllocator(...)`
- memory pattern controls and execution-plan integration

This is the primary place where runtime code resolves “which allocator should this tensor use?”

When debugging placement issues, verify allocator lookup in session state before investigating kernel logic.

## Data Transfer Responsibilities

In [onnxruntime/core/framework/data_transfer_manager.h](../../onnxruntime/core/framework/data_transfer_manager.h), `DataTransferManager` coordinates registered `IDataTransfer` implementations and exposes:

- synchronous and async tensor copy operations
- sparse tensor copy paths
- source/destination device routing

In [onnxruntime/core/framework/data_transfer_utils.h](../../onnxruntime/core/framework/data_transfer_utils.h), helper utilities wrap safe tensor copies to raw spans and typed spans.

If tensors are on different devices, correctness depends on this layer, not just allocator correctness.

## Runtime Flow Touchpoints

From session lifecycle code in [onnxruntime/core/session/inference_session.cc](../../onnxruntime/core/session/inference_session.cc):

- session options include memory pattern controls
- `FinalizeSessionState(...)` builds execution-time allocation state
- memory-pattern flags are resolved across graph and subgraphs
- allocator retrieval happens during feed/fetch and run paths

This is why memory bugs can appear only after initialization completes, even when load succeeded.

## Practical Debugging Path

Use this sequence for memory or copy issues:

1. confirm source and destination tensor devices and memory info
2. confirm allocator lookup through session state
3. confirm the expected data transfer implementation exists for that device pair
4. reproduce with minimal I/O binding or memcpy-style tests
5. only then inspect kernel-side assumptions

Do not start by changing kernel code when the failure mode suggests allocator or transfer misrouting.

## Test References

Use these tests as concrete patterns:

- [onnxruntime/test/providers/memcpy_test.cc](../../onnxruntime/test/providers/memcpy_test.cc): session-state allocator and copy path validation
- [onnxruntime/test/providers/io_binding_test.cc](../../onnxruntime/test/providers/io_binding_test.cc): binding inputs/outputs across CPU/GPU and explicit transfer behavior

For provider work, add targeted tests that cover:

- expected allocator selection for outputs
- CPU <-> device transfer correctness
- preallocated output behavior
- fallback and synchronization semantics

## Fast Validation Loop

From build output directory, run focused tests first:

Linux:

```bash
./onnxruntime_test_all --gtest_filter="*MemcpyTest*:*IOBinding*"
```

Windows:

```powershell
.\onnxruntime_test_all.exe --gtest_filter="*MemcpyTest*:*IOBinding*"
```

Narrow further to your exact test names while iterating.

## Design Rules

- keep allocator and transfer concerns separate from kernel math logic
- choose allocator by explicit device and memory info, not by assumptions
- only advertise provider memory capabilities you can satisfy
- make stream and synchronization behavior explicit in tests
- keep first fixes minimal and observable

## Common Failure Modes

- allocating on CPU and assuming implicit device transfer will happen
- using the wrong memory info for bound outputs
- missing or incorrect transfer registration for a device pair
- exposing stream-aware behavior without synchronization guarantees
- debugging execution outputs without first validating copy semantics

## Exit Checklist

- [ ] You can identify allocator selection points for the failing path.
- [ ] You can identify source and destination devices for each tensor copy.
- [ ] You validated copy and binding behavior with focused tests.
- [ ] Your change preserves allocator and transfer contracts across providers.