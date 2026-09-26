# WebGPU MatMul Algorithm Scheduler Design

## Goal

Make every `ComputeMatMul` implementation path explicit and independently testable. Selection policy must be separated from execution, retain the existing common rules as a fallback, and allow vendor-specific policy to override any performance heuristic.

## Algorithms

Introduce `MatMulAlgorithm` with five concrete values:

- `SubgroupMatrix`: the common subgroup-matrix implementation in `subgroup_matrix_matmul.cc`.
- `Naive`: `MatMulNaiveProgram`.
- `IntelSubgroup`: the Intel subgroup implementation in `vendor/intel/math/matmul.cc`.
- `Packed`: the generic packed `MatMulProgram` without Split-K.
- `PackedSplitK`: the same packed program with Split-K initialization and atomic accumulation.

There is no `Auto` algorithm value. Automatic versus forced selection is represented by an optional forced algorithm, which prevents an unset policy state from being confused with an executable implementation.

## Selection Architecture

Add a `MatMulAlgorithmScheduler` base class. Automatic selection uses this order:

1. Select `Naive` when `K == 0`. This is a correctness rule and cannot be overridden by vendor policy.
2. Ask the virtual `SelectVendorAlgorithm` policy hook for any algorithm. A vendor may use its own thresholds for every algorithm, not only vendor-specific implementations.
3. If the vendor returns no selection, call the private, non-virtual `SelectCommonAlgorithm` fallback. It selects `SubgroupMatrix` when supported, `Naive` when `N < 8 && K < 8`, `PackedSplitK` when the existing `SplitKConfig::UseSplitK` rule succeeds, and otherwise `Packed`.

An Intel-derived scheduler implements the vendor hook. It preserves the original policy by selecting `SubgroupMatrix` first when applicable, then selecting `IntelSubgroup` under the current Intel subgroup rule. Other vendors use the base scheduler unchanged. Future vendor policies can derive from the base scheduler, override as many performance ranges as needed, and return no selection to delegate the remaining ranges to the common fallback without adding vendor conditionals to `ComputeMatMul`.

The scheduler accepts immutable problem facts rather than mutable policy decisions: logical and packed dimensions, batch sizes, adapter architecture, input types, packing and layout facts, deterministic-compute state, activation and bias facts, and device capabilities. It also owns a copy of the immutable `SplitKConfig` selected for the adapter. The private common fallback evaluates `SplitKConfig::UseSplitK` directly from the problem facts instead of receiving a precomputed decision from `ComputeMatMul`. A vendor may ignore that common recommendation and apply independent thresholds from the raw facts. This keeps the scheduler deterministic and unit-testable without a WebGPU device.

`SplitKConfig` contains only generic Split-K eligibility evaluation and data. Adapter routing is handled by a small generic factory, while Intel architecture profiles and their measured threshold tables live under `vendor/intel`. `WebGpuContext` owns the selected configuration so GEMM and MatMul use the same profile; the MatMul scheduler receives that configuration when it is created. A future vendor can add its own profile builder and factory route without adding conditions to `ComputeMatMul` or changing the generic evaluator.

## Execution Configuration

Algorithm selection and execution tuning are separate decisions. After selecting one enum, the scheduler creates a `MatMulExecutionPlan` containing that enum and a typed algorithm configuration. It first asks the protected `SelectVendorConfiguration` hook for tuning, then uses private common defaults when the vendor declines. The tuning hook runs for both automatic and forced algorithms, so the test-only forcing option controls the implementation path without disabling real device tuning.

The packed configuration initially contains workgroup size, elements per thread, inner tile size, and Split-K size. `ApplyMatMulPacked` consumes those values directly and includes shader-affecting values in its cache key. The common configuration preserves the existing `8x8x1` workgroup, `4x1x1` or `4x4x1` elements-per-thread rule, inner tile size 32, and adapter Split-K size. A vendor may replace any of these values without changing `ComputeMatMul` or the packed implementation.

Packed tuning is validated at the execution boundary. The current shader requires both Z workgroup dimensions to remain one because Z identifies a batch or Split-K slice rather than a tiled output axis. Split-K sizes must be greater than one and aligned to the inner tile so adjacent workgroups cannot overlap their K ranges. Dispatch counts use widened, overflow-safe arithmetic and are range-checked before conversion to WebGPU's 32-bit dimensions.

Configuration is represented by an algorithm-specific variant rather than a bag of unrelated optional fields. The dispatcher rejects an algorithm/configuration type mismatch before execution. Empty configuration types reserve the same typed boundary for algorithms whose tuning remains inside their existing implementation. In particular, subgroup-matrix MatMul already receives a vendor-specific `SubgroupMatrixTilingSelector`; that existing selector continues to choose tile M, tile N, and split K without coupling the scheduler to device or shader classes.

## Forced Test Selection

Add the internal WebGPU session configuration key `ep.webgpuexecutionprovider.forceMatmulAlgorithm`. Accepted values are `subgroup_matrix`, `naive`, `intel_subgroup`, `packed`, and `packed_split_k`. The option is parsed when the WebGPU EP is created, stored as `std::optional<MatMulAlgorithm>`, and exposed read-only through `ComputeContextBase`.

When set, the scheduler returns the requested enum before applying heuristic rules. The dispatcher then validates the algorithm's hard prerequisites. Unsupported device features, data types, layouts, deterministic-compute settings, or other correctness constraints produce a descriptive failure naming the forced algorithm; forced mode never silently falls back.

Heuristic thresholds are not hard prerequisites. For example, forcing Intel subgroup bypasses its current `M/N/K` performance thresholds while still requiring an Intel adapter with subgroup support. Forcing Split-K bypasses performance thresholds while still requiring a usable Split-K configuration, non-deterministic compute, compatible packing/activation, and supported bias layout.

Invalid option strings fail during WebGPU provider creation and list accepted values.

## Dispatch and Implementation Boundaries

Introduce `MatMulComputeDispatcher` as the single compute entry point below the MatMul, pointwise Conv, and contrib Attention kernels. Each kernel owns one dispatcher for its lifetime. The dispatcher lazily creates adapter-dependent state from the first compute context and then owns:

- one `MatMulAlgorithmScheduler`, which contains selection policy and immutable adapter tuning data; and
- an optional `SubgroupMatrixMatMulImpl`, which contains only the subgroup-matrix implementation's persistent device state, including cached padded constant weights.

The dispatcher does not store a current or previously selected algorithm. Selection occurs for every invocation because shapes, input properties, activation, and bias can differ between calls. The session's forced-test configuration, when present, participates in each selection without becoming mutable dispatcher state. The resulting `MatMulExecutionPlan` is local to that invocation.

For each call, `MatMulComputeDispatcher::Compute` computes shared shape and capability facts once, asks the scheduler for one execution plan, validates the selected algorithm's hard prerequisites and configuration type, and switches directly to the matching implementation. MatMul, pointwise Conv, and contrib Attention delegate through this same entry point rather than coordinating the scheduler and implementations themselves.

The scheduler remains pure policy: it creates plans but does not create device programs, cache tensor data, or execute kernels. `SubgroupMatrixMatMulImpl` is named for the implementation it owns and is reached only when the plan selects `SubgroupMatrix`; its applicability query is non-mutating, and its execution method does not communicate selection through a `handled` output. This removes trial execution as a dispatch mechanism.

The other implementations remain focused stateless functions or program builders unless they acquire persistent state in the future. The naive, Intel subgroup, packed, and packed Split-K paths therefore do not receive empty polymorphic wrapper classes. The generic packed helper takes an explicit Split-K mode and packed configuration; it does not re-run selection or tuning heuristics. WebGPU's existing program cache continues to own reusable compiled programs.

This replaces `MatMulOptImplCache` and the generic `MatMulOptImpl` interface. If another algorithm later needs persistent implementation state, the dispatcher can own a separately named backend for that algorithm without changing scheduler policy or pretending that one object represents whichever algorithm happened to be selected most recently.

## Compatibility

With no forcing option or vendor override, the common scheduler remains equivalent to the previous selection order. The MatMul, pointwise Conv, and contrib Attention operator APIs and model semantics do not change; only their internal MatMul compute ownership moves behind the dispatcher.

The option is intentionally internal and test-only: it is declared with WebGPU provider options for configuration plumbing but is not added to public user documentation.

## Testing

- Add device-independent scheduler unit tests covering every common branch, forced-over-vendor precedence, the zero-K correctness guard, vendor-over-common precedence, Intel override, default fallback, independent vendor thresholds, common packed defaults, vendor tuning of a forced algorithm, Split-K profile routing, and current Intel architecture boundaries.
- Add parser/configuration tests for every accepted value and invalid input.
- Add WebGPU MatMul tests that choose shapes which normally select a different path, force a compatible algorithm, and verify numerical output. Hardware-specific forced algorithms are tested only when their hard capabilities are present; strict-failure tests cover unsupported forced choices.
- Build Dawn and the WebGPU provider on Windows with the Vulkan backend enabled and D3D12 disabled. Tests explicitly request `dawnBackendType=Vulkan`.
- Verify the selected Vulkan adapter exposes subgroup size control, f16, and the cooperative/subgroup-matrix configuration required by the 8x16x16 kernel before claiming subgroup-matrix execution coverage.
- Run scheduler/parser tests and hardware-backed MatMul tests on the local Intel Arc Vulkan adapter. macOS-arm64 Metal CI remains additional cross-backend coverage; lavapipe is not used because it cannot execute MatMul reliably.
