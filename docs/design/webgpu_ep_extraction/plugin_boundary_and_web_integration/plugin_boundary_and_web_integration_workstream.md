# Workstream `plugin-boundary`: Plugin Boundary and Web/Wasm Integration

Status: Working plan

[WebGPU EP extraction overview](../webgpu_ep_extraction.md)

## Objective

Make the public plugin EP interface the only runtime boundary between ONNX Runtime (ORT) and the WebGPU EP.

The same WebGPU provider implementation must work in two linkage modes:

| Host | Linkage | Factory discovery |
| --- | --- | --- |
| Native ORT | Shared plugin library | Runtime symbol lookup |
| `onnxruntime-web` and other static hosts | Static library | Direct factory registration |

This workstream owns the generic ORT infrastructure and host integration needed to make those modes equivalent. It
does not own WebGPU kernels, Dawn, shader tooling, or the provider's standalone build and release system.

## Desired end state

- WebGPU implements `OrtEpFactory` and `OrtEp` once.
- Dynamic and static linkage differ only in factory discovery, symbol visibility, and lifetime wiring.
- No WebGPU path reaches into `IExecutionProvider`, `OpKernel`, `Tensor`, `KernelRegistry`, or other private ORT
  interfaces.
- `onnxruntime-web` registers the statically linked factory through a generic ORT facility.
- Browser objects cross a narrow, documented boundary with explicit ownership and lifetime rules.
- Native hosts can load WebGPU as an optional plugin without core ORT packages depending on it.
- The legacy direct/static WebGPU provider path is removed after parity is demonstrated.

## Generic static plugin registration

Add an ORT facility that accepts statically linked plugin factory entry points. It should reuse the existing dynamic
plugin path after library loading and symbol lookup.

The design must address:

- Unique internal entry-point names when multiple static plugins are linked.
- Factory, device, allocator, data-transfer, and process-global lifetimes.
- Registration timing relative to environment creation.
- Cleanup when there is no dynamic-library unload event.
- Reduced and extended-minimal builds.
- Dead-code elimination for statically linked providers.
- Diagnostics for incompatible or duplicate registrations.

The facility must be generic and validated with at least one non-WebGPU test plugin where practical.

## Process-global ownership and teardown

Static registration removes the dynamic-library unload boundary. The current WebGPU plugin cleanup path cannot be
reused unchanged: releasing a factory clears global WebGPU contexts and kernel registries, destroys a global logger
wrapper, and shuts down protobuf. In a statically linked process, those subsystems may still be used by ORT, another
factory, or another static plugin.

Before static WebGPU parity is considered complete, classify every process-global subsystem as:

- Factory-owned and safe to release with that factory.
- Provider-registration-owned and released after the last factory and session using that registration.
- Host-owned and never finalized by the provider.

The static registration and provider lifetime design must cover:

- Multiple factories and devices from one registration.
- Multiple static plugin registrations in one process.
- Duplicate registration and partial initialization failures.
- ORT environment and session teardown ordering.
- Browser worker and process-exit behavior.
- Reference counting where provider-global state is shared.
- Logging lifetime without invalidating the host logger.
- Protobuf lifetime without calling `ShutdownProtobufLibrary()` on host-owned state.
- WebGPU context and kernel-registry caches without invalidating live sessions or factories.

`ReleaseEpFactory` must release only state whose ownership and last-user condition are established. A prototype that
executes correctly but retains unsafe dynamic cleanup behavior does not satisfy static parity.

## WebGPU plugin-path parity

The shared-library plugin form already builds and ships. The deliverable here is the static form for Emscripten and
other static hosts, plus lifetime correctness in both. Exercise the same factory, device discovery, provider options,
allocator, data transfer, graph assignment, and execution code in each.

Parity work includes:

- Native shared-plugin execution.
- Native static registration as a focused test host where useful.
- Emscripten static registration.
- Provider option behavior.
- GPU tensor and buffer interoperability.
- Error and diagnostic behavior.
- Process, environment, factory, and session lifetime behavior.

The direct in-tree WebGPU provider path remains only as a temporary comparison baseline.

## Plugin API gap closure

Provider-isolation work will identify uses of private ORT interfaces. Each finding should be resolved by:

1. An existing public `OrtApi` or `OrtEpApi` operation.
2. A public plugin EP API addition when a confirmed gap represents a stable runtime boundary useful beyond WebGPU.
3. A WebGPU-owned helper or replacement in the `provider-isolation` workstream.

Stable API additions have a high compatibility cost. Convenience helpers and provider implementation details should
not be moved into ORT's public API merely to simplify extraction.

Likely investigation areas include:

- External-data loading in WebAssembly.
- Graph and model information needed during capability discovery or compilation.
- Device tensors and externally owned buffers.
- Reduced-operator configuration.
- Logging, threading, allocators, and data transfer.
- Environment and process-global initialization.
- Setting EP default configuration before a session exists, equivalent to the existing `SetCurrentGpuDeviceId`.

The adapter's own `Missing parts` section in `onnxruntime/core/providers/webgpu/ep/README.md` is authoritative input
to this inventory rather than speculation. It records two gaps: WebGPU cleanup, which the process-global ownership
and teardown work covers, and EP default configuration, which is missing for both static and shared library builds
and sketches an `OrtApi` addition for it.

## Browser/Wasm bridge

Define the smallest stable interface needed to pass JavaScript-owned WebGPU objects between ORT Web and the provider.

The ORT repository should retain generic Wasm module assembly, JavaScript package behavior, and ORT lifecycle wiring.
The provider repository should own WebGPU-specific behavior. The boundary must define:

- `GPUDevice` and `GPUBuffer` representation.
- Ownership, reference, and destruction rules.
- Threading and async behavior.
- Device-loss propagation.
- Validation and error reporting.
- Compatibility with JavaScript and Emscripten changes.

The bridge should not expose unrelated ORT private implementation details.

## Work packages

1. **Size and latency baselines:** measure `onnxruntime-web` WebAssembly size and inference latency before any
   plugin-path change lands, since the size and latency completion criteria are relative to them.
2. **Static registration core:** implement and contract-test generic static factory registration.
3. **Global lifetime contract:** inventory process-global state and implement safe ownership and teardown rules.
4. **Emscripten prototype:** compile the adapter path statically and run a small model.
5. **Gap inventory triage:** convert private-dependency findings into public API or provider-owned actions.
6. **Browser bridge:** specify and prototype object and lifetime exchange.
7. **Parity and retirement:** run the existing suite through the plugin path and remove the legacy path.

Baselines come first, because they cannot be captured once the plugin path has changed anything. Packages 2 through 6
can then proceed largely in parallel. Legacy-path removal waits for their convergence.

## Interfaces with other workstreams

### Provider isolation and repository migration

- The `provider-isolation` workstream supplies concrete private-dependency findings.
- This workstream supplies the public plugin EP API headers and static host-registration contract.
- The `provider-isolation` workstream produces the static and shared libraries consumed by parity tests.

### Test ownership and operator conformance

- The `test-conformance` workstream supplies blocking parity cases and fallback detection.
- This workstream provides dynamic and static registration hooks for conformance runners.
- Contract tests for generic plugin infrastructure remain in ORT.

## Completion criteria

- Static and dynamic WebGPU builds use the same provider implementation and public API boundary.
- Factory release and environment teardown cannot shut down process-global state still owned or used by ORT, another
  factory, or another plugin.
- Static WebGPU does not shut down host-owned protobuf or logging state.
- A WebGPU model executes through static registration in `onnxruntime-web`.
- Existing plugin-path tests are blocking and detect fallback.
- All required private-runtime interactions have a documented public API or provider-owned replacement.
- Browser object ownership and lifecycle are documented and tested.
- Reduced WebAssembly builds retain required plugin infrastructure within accepted size budgets, measured against
  the baselines established before the work begins.
- Inference latency through static plugin registration stays within an accepted tolerance of the same baselines.
- The direct `IExecutionProvider` WebGPU path is removed.

## Open questions

- Should static factories be registered before environment creation or through environment construction options?
- What is the stable representation of JavaScript-owned WebGPU objects at the C API boundary?
- Which private-dependency findings require a public plugin EP API addition rather than a provider-owned replacement?
