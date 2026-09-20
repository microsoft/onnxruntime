# WebGPU MatMul Algorithm Scheduler Design

## Goal

Make every `ComputeMatMul` implementation path explicit and independently testable without changing the default runtime behavior. Selection policy must be separated from execution, preserve the current rule order, and allow vendor-specific policy to extend the common rules.

## Algorithms

Introduce `MatMulAlgorithm` with five concrete values:

- `SubgroupMatrix`: the common subgroup-matrix implementation in `subgroup_matrix_matmul.cc`.
- `Naive`: `MatMulNaiveProgram`.
- `IntelSubgroup`: the Intel subgroup implementation in `vendor/intel/math/matmul.cc`.
- `Packed`: the generic packed `MatMulProgram` without Split-K.
- `PackedSplitK`: the same packed program with Split-K initialization and atomic accumulation.

There is no `Auto` algorithm value. Automatic versus forced selection is represented by an optional forced algorithm, which prevents an unset policy state from being confused with an executable implementation.

## Selection Architecture

Add a `MatMulAlgorithmScheduler` base class. Its common rule order mirrors the existing `ComputeMatMul` condition order:

1. Select `SubgroupMatrix` when its implementation reports that it can handle the problem.
2. Select `Naive` when `N < 8 && K < 8`.
3. Ask a virtual vendor-policy hook for a vendor algorithm.
4. Select `PackedSplitK` when the existing `SplitKConfig::UseSplitK` rule succeeds.
5. Fall back to `Packed`.

An Intel-derived scheduler implements the vendor hook and selects `IntelSubgroup` under the current Intel subgroup rule. Other vendors use the base scheduler unchanged. Future vendor policies can derive from the base scheduler without adding vendor conditionals to `ComputeMatMul`.

The scheduler accepts already-computed selection facts rather than owning tensor execution. This keeps it deterministic and unit-testable without a WebGPU device. Vendor policy may inspect dimensions and device-derived capability facts, but it must return only a `MatMulAlgorithm`.

## Forced Test Selection

Add the internal WebGPU session configuration key `ep.webgpuexecutionprovider.forceMatmulAlgorithm`. Accepted values are `subgroup_matrix`, `naive`, `intel_subgroup`, `packed`, and `packed_split_k`. The option is parsed when the WebGPU EP is created, stored as `std::optional<MatMulAlgorithm>`, and exposed read-only through `ComputeContextBase`.

When set, the scheduler returns the requested enum before applying heuristic rules. The dispatcher then validates the algorithm's hard prerequisites. Unsupported device features, data types, layouts, deterministic-compute settings, or other correctness constraints produce a descriptive failure naming the forced algorithm; forced mode never silently falls back.

Heuristic thresholds are not hard prerequisites. For example, forcing Intel subgroup bypasses its current `M/N/K` performance thresholds while still requiring an Intel adapter with subgroup support. Forcing Split-K bypasses performance thresholds while still requiring a usable Split-K configuration, non-deterministic compute, compatible packing/activation, and supported bias layout.

Invalid option strings fail during WebGPU provider creation and list accepted values.

## Dispatch and Implementation Boundaries

`ComputeMatMul` computes shared shape facts once, asks the scheduler for exactly one enum, and switches directly to the matching implementation. Algorithm bodies are extracted into focused helpers where necessary. The generic packed helper takes an explicit Split-K mode; it does not re-run the selection heuristic.

The subgroup-matrix optional implementation gains a non-mutating applicability query and an execution method that no longer communicates selection through a `handled` output. This removes trial execution as a dispatch mechanism.

The existing per-kernel cache continues to own device-dependent subgroup-matrix state and the scheduler, so MatMul, pointwise Conv, and Attention callers retain their current caching and behavior.

## Compatibility

With no forcing option, the selected algorithm and precedence remain equivalent to the current code. Existing call sites keep using `ComputeMatMul`; the refactor does not change the operator API or model semantics.

The option is intentionally internal and test-only: it is declared with WebGPU provider options for configuration plumbing but is not added to public user documentation.

## Testing

- Add device-independent scheduler unit tests covering every common branch, precedence, Intel override, default fallback, and forced override.
- Add parser/configuration tests for every accepted value and invalid input.
- Add WebGPU MatMul tests that choose shapes which normally select a different path, force a compatible algorithm, and verify numerical output. Hardware-specific forced algorithms are tested only when their hard capabilities are present; strict-failure tests cover unsupported forced choices.
- Build Dawn and the WebGPU provider on Windows with the Vulkan backend enabled and D3D12 disabled. Tests explicitly request `dawnBackendType=Vulkan`.
- Verify the selected Vulkan adapter exposes subgroup size control, f16, and the cooperative/subgroup-matrix configuration required by the 8x16x16 kernel before claiming subgroup-matrix execution coverage.
- Run scheduler/parser tests and hardware-backed MatMul tests on the local Intel Arc Vulkan adapter. macOS-arm64 Metal CI remains additional cross-backend coverage; lavapipe is not used because it cannot execute MatMul reliably.
