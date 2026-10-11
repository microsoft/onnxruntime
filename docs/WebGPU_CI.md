# WebGPU test coverage in Windows CI

The shared plugin job owns the full provider numerical suite. Static-plugin and
built-in WebGPU jobs retain tests of their integration paths. This avoids repeating
the expensive operator sweeps while keeping coverage of the supported build modes.

| Configuration | Provider tests | Other coverage retained |
|---|---|---|
| Shared plugin | Full shared operator suite and WebGPU-specific operator tests | Plugin loading, allocators, concurrency, recording, framework/device selection; separate plugin implementation tests; Java plugin registration/inference |
| Static plugin | Focused plugin allocator, concurrency and recording tests | WebGPU framework and automatic device-selection tests; registration/linking into the host |
| Built-in WebGPU, vcpkg and no-vcpkg | Context, allocator, transfer and concurrency tests plus HardSwish smoke cases | D3D12 accelerated weight loading, built-in factory/framework selection, existing language-binding and DLL-loading checks |

The built-in configuration is still required: D3D12 accelerated weight loading is
not currently supported by plugin builds. Its tests cannot be removed on the basis
of plugin numerical coverage. External-Dawn and package-backed host tests also
remain in their existing configurations.

## Plugin implementation tests

Operator tests that only need public EP functionality run in
`onnxruntime_provider_test`. Tests of private context, dispatch, buffer and shader
helpers run in `onnxruntime_webgpu_internal_test`, which loads a test-only module
containing the same compiled provider objects used by the shared plugin.
The production plugin's exported interface is unchanged.

Some operator regressions need private setup: a smaller storage-binding limit,
a forced MatMul algorithm, or an exact capability/counter query. The test helper
uses the private module for these controls, then executes models through the
public plugin interfaces. It does not add production provider options for tests.

The host supplies real kernel-info objects through the public API for direct
program tests. Test data, shapes, expected results and numerical tolerances remain
in the original tests. A relocated test must be registered exactly once across
the operator and implementation test executables.

## Modern and serialized plugin execution

`onnxruntime_webgpu_legacy_test` exercises the plugin's compatibility recording
mode using `ORT_WEBGPU_EP_FORCE_LEGACY=1`. Its name does not mean the built-in EP.
These compatibility checks remain in shared and static plugin configurations.

`onnxruntime_webgpu_internal_serialized_test` preserves the deferred allocator-clear
contract in the implementation tests. Modern plugin streamless allocation clears
immediately; the normal internal test run checks that behavior separately.

## Selecting focused integration coverage

Set `onnxruntime_WEBGPU_INTEGRATION_TESTS_ONLY=ON` for the focused Windows jobs.
For plugin builds, the provider binary includes only the plugin integration sources.
For built-in builds, CTest filters the provider binary to the integration/smoke set;
the full binary remains available for explicit local diagnostics.
CTest also filters framework and automatic-device-selection tests to WebGPU cases,
retaining the D3D12 external-loader lifecycle tests in the built-in configuration.

The static-plugin workflow explicitly runs the provider, compatibility-recording,
framework and automatic-device-selection CTests. Zero matched CTests and zero
matched GTests are errors. The existing Windows required gate also checks the full
shared-plugin job's result, including failure, cancellation and skipped execution.

This change does not alter the existing macOS Metal or Linux build configurations.
Their platform coverage is retained; local Windows results do not validate Metal,
Vulkan or other execution providers.

## Coverage verification

Before changing test ownership, record inventories including disabled tests for
the original built-in and shared-plugin binaries. Compare that union with the
shared-plugin operator and implementation inventories, requiring no missing
identities and no overlap. Compare actual pass/skip results for migrated cases,
and investigate new skips or tests that pass without executing their intended EP.
Factory/allocator tests moved to plugin-specific forms need an explicit behavior
mapping in addition to the identity check.

GTest success alone is insufficient evidence: a shared operator helper can decline
an unsupported model/provider combination without failing its enclosing case.
Preserve provider options, CPU-fallback settings, private boundary controls and
capability predicates when adapting a test.
