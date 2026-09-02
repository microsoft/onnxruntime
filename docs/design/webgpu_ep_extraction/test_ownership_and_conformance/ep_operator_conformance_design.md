# Execution Provider Operator Conformance Suite

Status: Detailed design supporting
[Test Ownership and Operator Conformance](test_ownership_and_conformance_workstream.md).

## Purpose

Define a reusable ONNX Runtime (ORT) operator conformance suite that can validate in-tree and external execution
providers (EPs) against common operator semantics.

The suite should let an EP author test a released plugin without cloning or building the ORT repository. It should
also preserve the value of ORT's existing operator tests while separating portable test cases from ORT-private C++
test infrastructure.

This facility is general to all EPs. WebGPU is an initial consumer and a useful migration test case, not a special
case in the design.

## Background

PR [#25689](https://github.com/microsoft/onnxruntime/pull/25689) created `onnxruntime_provider_test`, moved provider
and operator tests into that executable, and allowed tests using `OpTester` or `ModelTester` to run with a dynamically
registered plugin EP.

That established two useful foundations:

- Provider tests can run separately from the main ORT unit-test executable.
- An EP can be selected at runtime instead of being statically known to each test.

`onnxruntime_provider_test` remains an in-tree ORT test binary. Its tests link private framework, graph, optimizer,
provider, and test libraries, and many cases are expressed as compiled C++ code. External EP repositories cannot
consume it as a stable release interface.

## Goals

- Provide common operator correctness cases that can be run against any EP.
- Use public ORT APIs at the runner boundary.
- Prevent CPU fallback from producing false passes.
- Support dynamically loaded and statically linked plugin EPs with the same cases and result rules.
- Support native and browser/Wasm runners without changing case meaning.
- Publish a versioned conformance kit associated with an ORT release.
- Make unsupported cases, exclusions, and tolerance overrides explicit and reportable.
- Allow existing `OpTester` and `ModelTester` cases to migrate incrementally.

## Non-goals

- Replace all tests in `onnxruntime_provider_test`.
- Make `OpTester`, `ModelTester`, or other ORT-private test helpers a stable public C++ API.
- Validate provider-specific implementation details such as shaders, vendor libraries, caches, or device limits.
- Require every EP to implement every operator in the suite.
- Define ONNX operator semantics independently from the ONNX specification.

## Proposed components

The design separates case authorship, case distribution, and execution:

1. **Case definitions** describe models, inputs, expected outputs, comparison rules, and requirements.
2. **Case generator** validates definitions and emits portable serialized models and datasets.
3. **Conformance kit** packages generated cases, schemas, documentation, and a native runner for an ORT release.
4. **Runner** registers an EP, executes selected cases, enforces fallback rules, and writes a structured report.
5. **Provider profile** declares the EP's expected support surface, options, exclusions, and narrow comparison
   overrides.

The portable contract is the case format and execution/result semantics. A particular runner implementation is not
the contract, and the vehicle for the runner is still open — see Open questions.

## Relationship to `onnxruntime_provider_test`

`onnxruntime_provider_test` should remain the comprehensive in-tree provider regression executable. It can use
ORT-private helpers and cover implementation details that are inappropriate for an external contract.

The conformance suite should be a second layer:

| Layer | Purpose | Dependencies |
| --- | --- | --- |
| `onnxruntime_provider_test` | In-tree provider and operator regression testing | ORT-private libraries and helpers |
| `onnxruntime_ep_conformance_test` | Portable EP operator conformance | Public ORT APIs and released case data |

During migration, an `OpTester` or `ModelTester` case may remain the authoring source while tooling exports an
equivalent portable case. Over time, reusable cases should have one canonical data-driven definition consumed by
both test layers where practical.

## Case ownership and storage

Canonical ORT and contrib-op case definitions should live in the ORT repository, for example:

```text
onnxruntime/test/ep_conformance/
  cases/
    onnx/
    contrib/
  schemas/
    case.schema.json
    provider-profile.schema.json
    report.schema.json
  tools/
    generate_cases.py
  runner/
```

Standard ONNX cases should reuse or derive from ONNX backend test data where practical. ORT owns cases for ORT contrib
operators and generic ORT EP behavior.

Generated `.onnx` models and large tensor datasets do not normally need to be checked in. ORT CI and release jobs can
generate them into a conformance-kit archive. A model should be checked in only when its exact serialized form is
part of the test or generation is not reasonably deterministic.

Provider repositories own their profiles, exclusions, and implementation-specific tests. For example:

```text
webgpu-ep/tests/conformance/
  provider-profile.json
  exclusions.json
```

## Case representation

A simple single-operator case can use a compact declarative representation containing:

- Stable case identifier.
- Operator domain, name, and opset version.
- Input and output names, types, shapes, and values.
- Operator attributes.
- Comparison policy.
- Required capabilities or environmental constraints.
- Execution requirements such as complete assignment to the target EP.

For example:

```json
{
  "id": "ai.onnx.Add.opset14.float32.broadcast",
  "operator": {
    "domain": "",
    "type": "Add",
    "opset": 14
  },
  "inputs": [
    {"name": "A", "type": "float32", "shape": [2, 3], "values": [1, 2, 3, 4, 5, 6]},
    {"name": "B", "type": "float32", "shape": [3], "values": [10, 20, 30]}
  ],
  "outputs": [
    {"name": "C", "type": "float32", "shape": [2, 3], "values": [11, 22, 33, 14, 25, 36]}
  ],
  "comparison": {
    "rtol": 0.0001,
    "atol": 0.00001,
    "nan_equal": true
  },
  "execution": {
    "require_target_ep": true,
    "allow_cpu_fallback": false
  }
}
```

The release generator wraps such a case in a valid ONNX model. The runner passes serialized model bytes to a normal
ORT session, so the test exercises graph capability discovery and compilation as well as execution.

Complex graphs, control flow, functions, external data, malformed models, and tests where exact protobuf structure
matters may use packaged `.onnx` models directly. Large or shared tensor values may use ONNX backend-test-style
`test_data_set_*` directories rather than inline values.

## Execution semantics

For an ordinary operator conformance case, the runner should:

1. Load or register the requested plugin EP and select a device.
2. Apply the requested EP options.
3. Create a session with CPU fallback disabled, unless the CPU EP is itself the target.
4. Load the generated or packaged model.
5. Require the target EP to accept the nodes specified by the case.
6. Run every input dataset.
7. Compare every output according to the case's comparison policy.
8. Record diagnostics without changing the defined outcome.

Disabling CPU fallback is essential. A correct result produced by the CPU EP does not demonstrate conformance of the
target EP. The exception is the CPU reference run, where the CPU EP is the target: there is nothing to fall back
from, and ORT rejects a session that disables CPU fallback while nodes are assigned to the CPU EP.

Cases that intentionally test partial graph assignment must state their assignment requirements explicitly. They
should be classified separately from single-operator correctness cases.

## Result semantics

Each selected case should produce exactly one result:

- `PASS`: The target EP executed the required graph and all outputs matched.
- `UNSUPPORTED`: The EP did not claim a case outside its declared support profile.
- `FAIL`: Output mismatch, crash, timeout, unexpected rejection, or unexpected fallback.
- `EXCLUDED`: A provider exclusion matched the case and supplied a documented reason.
- `NOT_RUN`: The environment could not execute the case, such as when no compatible device was available.

`UNSUPPORTED` is not automatically a passing result. If the provider profile says the case is supported, rejection
is a failure. `NOT_RUN` should make a required CI lane incomplete rather than successful.

An exclusion should identify a stable case ID, a reason, and preferably a tracking issue. Broad wildcard skip lists
should be discouraged because they obscure coverage loss.

## Comparison semantics

The case owns the default comparison policy. The policy may specify:

- Exact comparison.
- Absolute and relative tolerances.
- NaN and infinity handling.
- Type-specific rules.
- Ordering rules where the operator permits multiple valid orders.

A provider may define a narrow override when implementation precision requires it. Every override should include a
reason and should be visible in the report. Provider-wide tolerance inflation should not be supported.

## Provider profile

A provider profile defines test expectations rather than replacing the EP's runtime capability implementation. It
may contain:

- Plugin registration name and selected EP name.
- Device selection and EP options.
- Supported domains, opsets, data types, and optional features.
- Case tags to include or exclude from a particular environment.
- Documented exclusions.
- Narrow comparison overrides.

The runner uses the profile to distinguish an expected lack of support from a regression in the provider's declared
support surface.

## Distribution

An ORT release should publish a versioned conformance kit, for example:

```text
onnxruntime-ep-conformance-<ORT_VERSION>/
  bin/
    onnxruntime_ep_conformance_test
  cases/
    onnx/
    contrib/
  schemas/
  examples/
  VERSION
```

The native runner is platform-specific. The generated case archive and schemas should be platform-neutral.

An external EP should test against at least:

- The minimum ORT release it supports.
- The current ORT release used for packaging.
- An ORT `main` or nightly conformance kit as an early-warning lane.

## Dynamic plugin usage

A native dynamically loaded plugin could be tested as follows:

```powershell
onnxruntime_ep_conformance_test `
  --ep-library .\onnxruntime_providers_webgpu.dll `
  --registration-name webgpu_plugin `
  --ep-name WebGpuExecutionProvider `
  --cases .\cases `
  --provider-profile .\provider-profile.json `
  --report .\results.json
```

The registration name is chosen by the caller and identifies the loaded library. The EP name is the one the factory
reports, and selects which provider from that library to use. The runner should use public plugin registration,
device discovery, session creation, and execution APIs.

## Static plugin usage

A prebuilt executable cannot discover a statically linked plugin, so validating static linkage requires a runner the
EP repository builds itself. Whether that runner is needed is an open question below. If it is adopted, ORT would
also publish a small runner SDK or CMake target that allows an EP repository to supply static factory registration:

```cmake
find_package(onnxruntime_ep_conformance CONFIG REQUIRED)

add_executable(webgpu_ep_conformance static_ep_registration.cc)
target_link_libraries(
  webgpu_ep_conformance
  PRIVATE
    onnxruntime::ep_conformance_runner
    webgpu_ep_static
)
```

This executable should consume the same case archive and produce the same report as the dynamic runner. Static and
dynamic linkage must not create separate conformance definitions.

## WebAssembly and browser usage

`onnxruntime-web` cannot use the native dynamic-plugin executable. A JavaScript or browser runner should load the same
portable cases, invoke the statically registered WebGPU plugin through ORT Web, and produce results with the same
schema and outcome rules.

Browser-specific scheduling, test sharding, and artifact loading are host concerns. They must not change the meaning
of `PASS`, `FAIL`, `UNSUPPORTED`, `EXCLUDED`, or `NOT_RUN`.

## Report format

Reports should be machine-readable and include enough provenance to reproduce a run:

- Report schema version.
- Conformance-suite and ORT versions.
- EP name and version.
- Dynamic or static registration mode.
- Device and relevant environment information.
- Provider profile hash.
- Per-case result, duration, and diagnostics.
- Summary counts by result, domain, operator, opset, and data type.

The report should make newly unsupported or newly excluded cases easy to detect in CI.

## Migration approach

### Phase 1: Define the contract

- Define case, provider-profile, and report schemas.
- Implement a public-API-only native runner for dynamic plugin EPs.
- Enforce CPU-fallback prevention.
- Convert a small representative set of ONNX and contrib operators.
- Run those cases against CPU and at least one plugin EP.

### Phase 2: Connect existing test infrastructure

- Add an export path from suitable `OpTester` and `ModelTester` cases.
- Generate conformance cases in CI and verify deterministic output.
- Produce a coverage map from existing provider tests to conformance case IDs.
- Keep `onnxruntime_provider_test` authoritative until converted cases demonstrate parity.

### Phase 3: Publish and consume release kits

- Publish native runners and platform-neutral case archives with ORT releases.
- Add a nightly kit for ORT `main`.
- Add a static-runner SDK or CMake package, if a native static-linkage runner is adopted.
- Integrate the conformance kit into an external plugin EP repository.

### Phase 4: Expand coverage and hosts

- Migrate broadly reusable operator cases.
- Add browser/Wasm execution of the same cases.
- Add conformance coverage reporting to ORT and EP CI.
- Retain implementation-specific tests in their owning repositories.

## Initial success criteria

- One case definition runs against CPU and a dynamically loaded plugin EP.
- The same case detects and fails unexpected CPU fallback.
- Dynamic and static forms of one plugin produce equivalent results.
- An external EP repository can run a conformance kit without an ORT source checkout.
- Results distinguish failures, unsupported cases, exclusions, and infrastructure failures.
- Existing provider coverage can be mapped to stable conformance case IDs without an all-at-once migration.

## Open questions

- What should the runner be built on? Candidates are a new public-API-only C++ runner as sketched here,
  `onnx_test_runner` extended with plugin EP registration, or a Python suite over the existing plugin registration
  APIs. `onnx_test_runner` already ships, consumes ONNX backend test data, and supports disabling CPU fallback, but
  links private ORT libraries. A Python suite has no build barrier for external consumers but covers native dynamic
  loading only.
- Is a native static-linkage runner needed at all? The static-linkage consumer is `onnxruntime-web`, which requires a
  browser runner regardless, so a native static host may be a hypothetical consumer.
- Should canonical simple cases use JSON, protobuf, Python source, or another representation?
- Which ONNX backend cases can be consumed directly without duplication?
- What public mechanism best proves target-EP assignment when partial assignment is allowed?
- How should capability profiles express operator attributes and shape constraints without duplicating
  `GetCapability()`?
- Should contrib-op cases ship in the default kit or a separate ORT-extension bundle?
- Which runner artifacts should be included in each ORT package and release channel?
- How should tensor data shared across cases be deduplicated?
- How should large datasets be versioned and distributed?
- What compatibility promise applies to case, provider-profile, and report schema versions?
- How should randomized or generated inputs remain deterministic and reproducible?
- What is the minimum representative set of operators and data types needed before using the suite as an extraction
  prerequisite?
