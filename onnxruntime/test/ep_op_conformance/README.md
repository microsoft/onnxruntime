# EP Operator Conformance Pilot

This is a narrow native prototype, not complete generic-test migration, WebGPU parity, or a released conformance kit. The
[design](../../../docs/design/webgpu_ep_extraction/test_ownership_and_conformance/ep_operator_conformance_design.md)
and [workstream](../../../docs/design/webgpu_ep_extraction/test_ownership_and_conformance/test_ownership_and_conformance_workstream.md)
define the agreed architecture and future extraction gates. This README describes the implemented bounded pilot
and its draft contracts. Local revised checks pass; the new workflow has not been run on GitHub. EP-specific
unit-test migration and separate browser/Wasm continuity are out of scope.

The external audience is EP implementers. Planned distribution is versioned standalone runner source plus immutable
generated cases and contracts/provenance; prebuilt runners are optional. CI source-kit packaging and opt-in main ORT
build integration are implemented; release publication remains separate. The standalone commands below use this
checkout's public headers without building ORT; [source-kit instructions](SOURCE_KIT.md) use only packaged headers. The
[CI rollout](../../../docs/design/webgpu_ep_extraction/test_ownership_and_conformance/test_ownership_and_conformance_workstream.md#work-packages)
separates initial source-built CPU/source-kit validation from later provider-workflow adoption.

## Implementation

- [runner/main.cc](runner/main.cc): public-API-only native consumer, dynamic ORT runtime loading, target setup,
  assignment checks, execution, and JSON reporting.
- [runner/case_io.h](runner/case_io.h) and [runner/comparison.h](runner/comparison.h): shared artifact loader and
  float32 comparator used regardless of target.
- [tools/generate_cases.py](tools/generate_cases.py): Python/ONNX producer for independently authored data definitions.
- [cases/pilot.json](cases/pilot.json): scalar Mul, initializer Mul, and broadcast Add, once per dataset.
- [examples/cpu.json](examples/cpu.json), [examples/dml.json](examples/dml.json), and
  [examples/sample_plugin.json](examples/sample_plugin.json): CPU, legacy DirectML, and sample-plugin profiles.
- [../unittest_util/ep_op_conformance_export.cc](../unittest_util/ep_op_conformance_export.cc) and
  [../unittest_util/base_tester.cc](../unittest_util/base_tester.cc): ORT-private C++ producer.
- [../providers/cpu/ep_op_conformance_export_test.cc](../providers/cpu/ep_op_conformance_export_test.cc): CPU-authored Mul
  export fixture and rejection tests, built as `onnxruntime_ep_op_conformance_export_test`.

The standalone C++20 runner uses public ORT headers, ONNX protobuf, and nlohmann JSON. It loads the supplied ORT shared
library dynamically, requests API version 24, and rejects runtimes without that API. ORT 1.24 is the minimum for the
public graph-assignment APIs, not a guarantee that every model/opset/provider is supported. No private ORT libraries
or Python are required at runner execution time. Python is used only for generation/testing.

## Supported slice

The pilot handles named inline dense float32 tensors, standard-domain successful cases, authored expected outputs,
initializers versus feeds, scalar shapes, explicit per-output comparison policies, and **one execution per dataset**
on the case's session. Neither authoring JSON nor the generated manifest accepts `repeat_count`. Export rejects
`num_run_calls_ != 1`, rather than silently reducing a C++ test to one run.

Actual output dtype, rank, dimensions, and element count must match exactly; there is no broadcasting or dtype
coercion. Finite values pass using double-precision comparison arithmetic when

$$
|a-e| \leq \mathrm{atol} + \mathrm{rtol}\,|e|
$$

where $a$ is actual and $e$ is authored expected. Tolerances are finite and nonnegative; zero gives exact numeric
equality, not byte equality (positive and negative zero match). NaNs match only when both are NaN and `nan_equal` is
true. Infinities match only with the same sign; lone NaNs and finite/infinite pairs always fail. JSON generation
accepts finite values representable as finite float32 only, even though TensorProto comparison handles NaNs/infinities.

The JSON generator is bounded to one standard-domain node without attributes and one dataset. Concrete dimensions,
unique tensor names, and exact element/byte counts are required. Tensor files are inline dense float32, with
`float_data` or little-endian `raw_data`; external data, segments, sparse/container/optional values, and other dtypes
are unsupported. Export additionally rejects nonstandard domains, functions/training graphs, nested/sparse graph
attributes, and unsupported assertions/settings. Reject unsupported forms instead of dropping semantics.

Sessions disable graph optimizations, enable `session.record_ep_graph_assignment_info`, and require every named
`required_nodes` entry on the profile's `ep_name`. Every reported assignment must also belong to that target.
Non-CPU targets set `session.disable_cpu_ep_fallback=1`; CPU does not, because CPU itself is the target. Assignment
evidence plus successful execution is not proof of an EP's internal hardware execution.

Built-in (including CPU) and plugin targets share artifact bytes, loading, session protocol, comparison, and report
logic. Setup adapters differ. CUDA V2 and DirectML's public append symbol are internal compatibility details; generic
append uses the full canonical `ep_name` where supported. This is not universal EP support: a compatible
runtime/provider build and its native dependencies are still necessary. A provider-bridge DLL is not automatically
a plugin API library.

Not delivered: full generic-test migration/WebGPU parity, complete coverage accounting, artifact-to-JSON conversion,
custom portable validators, FP16/other value types, expected-error cases, arbitrary session settings, profile tolerance overrides,
release packaging, real WebGPU validation, browser/Wasm, or a native static host/SDK.

## Case ID convention

Use a stable, provider-neutral ID with an operator-identifying prefix and a descriptive variant:

```text
<domain>.<operator>.opset<version>.<descriptive_variant>
```

Use `ai.onnx` for the standard ONNX domain and the declared domain for custom operators. Preserve the operator
schema's capitalization and use the model's imported opset version. Custom-domain examples describe the naming
convention, not additional support in the bounded generator.

The variant describes the behavior being exercised, including types, shapes, attributes, or initializer roles when
useful. There is no mandatory dtype slot: operators may have multiple input/output types. Prefer lowercase
`snake_case` for new variants, for example `ai.onnx.Cast.opset19.float32_to_int64` or
`ai.onnx.TopK.opset11.largest_sorted`. The final variant contains only ASCII letters, digits, and underscores;
dots separate components. For example, use `ai.onnx.Mul.opset14.float32_scalar`.

IDs are used in manifests, reports, and exact-match skip profiles. They must be unique across the selected corpus
and remain stable across source reorderings. Do not include EP names, platforms, outcomes, tolerance values, or
content hashes. The model and tensors are authoritative; consumers must not infer types or graph semantics by
parsing the ID. Avoid cosmetic renames once an ID is referenced.

The generator stores each case in a `case-<id>` directory, for example
`case-ai.onnx.Mul.opset14.float32_scalar`. The manifest, reports, and skip profiles retain the unprefixed logical ID.
The `case-` prefix avoids some invalid filename cases.
The generator enforces these portability rules:

- Follow the structured ID above. Domain names use ASCII letters, digits, `.`, `_`, and `-`; the operator uses
  ASCII letters, digits, and `_`; the opset is a positive decimal integer without leading zeros; the final variant
  uses ASCII letters, digits, and `_`. Every component is nonempty.
- Do not create IDs differing only by ASCII letter case, since their directories collide on case-insensitive filesystems.

Keep IDs and output paths reasonably short to fit the host's filesystem limits. The runner reads identity from the manifest;
it does not require an ID-derived directory name, and the C++ export API still accepts an explicit directory path.

## Build the standalone runner

[CMakeLists.txt](CMakeLists.txt) is independent of the main ORT build. It requires CMake 3.26+, a C++20 toolchain, public
ORT headers exposing the required APIs, nlohmann JSON headers, and matching generated ONNX protobuf headers/libraries
and protobuf headers/library. Match architecture, configuration, and C++ runtime across dependencies. The runtime
library is selected later by `--runtime`, not linked here.

The simplest build fetches hash-pinned public dependencies: ONNX 1.17.0, protobuf 3.21.12, and nlohmann JSON 3.11.3.
It uses this checkout's public ORT headers by default, not its private libraries. Python is needed during the ONNX
dependency build; it is not linked into the runner. These PowerShell examples start at the repository root:

```powershell
$Repo = (Get-Location).Path
$PilotBuild = Join-Path $Repo 'build/ep_op_conformance/native'
cmake -S "$Repo/onnxruntime/test/ep_op_conformance" -B "$PilotBuild" `
  -G 'Visual Studio 17 2022' -A x64 -DCONFORMANCE_FETCH_DEPS=ON `
  "-DPYTHON_EXECUTABLE=$Repo/.venv/Scripts/python.exe"
cmake --build "$PilotBuild" --config Release --target onnxruntime_ep_op_conformance_test ep_op_conformance_comparison_test
ctest --test-dir "$PilotBuild" -C Release --output-on-failure -R '^ep_op_conformance_comparison$'
```

Reuse the ORT build/test Python environment described below before configuring. Alternatively, supply all six cache paths
with `EP_OP_CONFORMANCE_FETCH_DEPS=OFF` (the default) in a separate build directory. Replace paths with your SDK/build paths;
use an appropriate toolchain/generator for your platform. On other platforms substitute native executable and
shared-library filenames. A single-configuration generator also needs `-DCMAKE_BUILD_TYPE=Release`.

```powershell
$Repo = (Get-Location).Path
$PilotBuild = Join-Path $Repo 'build/ep_op_conformance/native'
cmake -S "$Repo/onnxruntime/test/ep_op_conformance" -B "$PilotBuild" `
  "-DORT_PUBLIC_INCLUDE_DIR=$Repo/include/onnxruntime/core/session" `
  '-DJSON_INCLUDE_DIR=C:/deps/json/include' `
  '-DONNX_PROTO_INCLUDE_DIR=C:/deps/onnx-generated' `
  '-DPROTOBUF_INCLUDE_DIR=C:/deps/protobuf/include' `
  '-DONNX_PROTO_LIBRARY=C:/deps/lib/onnx_proto.lib' `
  '-DPROTOBUF_LIBRARY=C:/deps/lib/libprotobuf.lib'
cmake --build "$PilotBuild" --config Release --target onnxruntime_ep_op_conformance_test ep_op_conformance_comparison_test
ctest --test-dir "$PilotBuild" -C Release --output-on-failure -R '^ep_op_conformance_comparison$'
```

All six cache paths are required in manual-dependency mode. `ORT_PUBLIC_INCLUDE_DIR` contains `onnxruntime_cxx_api.h`; `JSON_INCLUDE_DIR` contains
`nlohmann/json.hpp`; `ONNX_PROTO_INCLUDE_DIR` contains generated `onnx/onnx-ml.pb.h`. CMake also links the platform's
dynamic-loader library where needed. The registered CTest above checks the comparator, not full EP execution.

## Generate independent data cases

Reuse the ORT build/test Python environment for generation; no separate producer environment is required.
For generation without a source build, an ordinary development virtual environment with NumPy and ONNX is sufficient:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy onnx
$RunRoot = Join-Path $Repo ('build/ep_op_conformance/run-' + (Get-Date -Format 'yyyyMMdd-HHmmssfff'))
$DataCases = Join-Path $RunRoot 'data_cases'
python onnxruntime/test/ep_op_conformance/tools/generate_cases.py `
  --definitions onnxruntime/test/ep_op_conformance/cases/pilot.json --output "$DataCases"
```

Choose a fresh output directory each time. The generator refuses an existing destination. Each case contains
`case.json`, `model.onnx`, and named TensorProto input/expected-output files under `test_data_set_0/`. The model retains
initializers; they are not feeds. Expected values come from the authored definition, not from a CPU/plugin run.

The manifest carries schema version 1, stable ID, model path, required node names, dataset paths, per-output
`atol`/`rtol`/`nan_equal`, and source metadata. There is no run-count field. Treat generated files as immutable
derived outputs, not a second authored corpus. The loader also accepts the exporter's flat tensor-file layout.

## Draft v1 contracts

The [design notes](../../../docs/design/webgpu_ep_extraction/test_ownership_and_conformance/ep_operator_conformance_design.md#authoring-json-v1-revised-draft)
contain complete bounded field tables and JSON examples. These are draft contracts, not a formal JSON Schema implementation.

| Format | Required content |
| --- | --- |
| Authoring JSON | `schema_version: 1`, nonempty `cases`; each case has `id`, `operator`, `opset`, `inputs`, `outputs`, `comparison` |
| Authored tensor | `name`, concrete `shape`, `values`; optional input `initializer` boolean (default false); implicit float32 |
| Comparison | `atol`, `rtol`, `nan_equal`; authoring case policy expands to one policy per output |
| Generated manifest | `schema_version: 1`, `id`, `model`, nonempty `required_nodes`/`datasets`, output-name `comparison` map, `source` provenance |
| Dataset | `inputs`/`outputs` arrays of relative TensorProto paths; exactly one execution, all expected model outputs required |
| Target profile | `schema_version: 1`, `kind: builtin` or `plugin`, canonical `ep_name`; optional string `options` map and reasoned exact-ID `skips` |
| Plugin additions | Required `library`, `registration_name`; optional nonnegative `device_index` (default zero) |

Unknown fields/versions fail, including old `repeat_count`, `kind: cpu`, and `provider_name` forms. Built-in profiles
cannot carry plugin setup fields. The canonical EP name is the sole provider identity; plugin library registration
has its own name. Case paths must stay inside the case directory, including resolved symlinks; absolute paths and
traversal fail. Missing/malformed artifacts, duplicate IDs/names, invalid policies, and unknown skips fail closed.

JSON is authoring input, not a runner input: `JSON -> generator -> model/tensors/manifest -> native runner -> report`.
The model retains graph/opset/type/initializer semantics, tensors retain named typed payloads, and the manifest
indexes paths, assignment requirements, comparisons, and provenance without duplicating values.

## Run CPU or a legacy EP

The CLI requires exactly these four option/value pairs. `--cases` accepts one case directory or a directory whose
immediate children contain case manifests; it is not a recursive search. The report's parent directory must exist.

```powershell
$Runner = Join-Path $PilotBuild 'Release/onnxruntime_ep_op_conformance_test.exe'
& $Runner --runtime C:/sdk/ort/onnxruntime.dll `
  --profile "$Repo/onnxruntime/test/ep_op_conformance/examples/cpu.json" `
  --cases "$DataCases" --report "$RunRoot/cpu.json"

& $Runner --runtime C:/sdk/ort-directml/onnxruntime.dll `
  --profile "$Repo/onnxruntime/test/ep_op_conformance/examples/dml.json" `
  --cases "$DataCases" --report "$RunRoot/dml.json"
```

Adjust `$Runner` for single-configuration build layouts. Ensure runtime/provider native dependencies are discoverable.
Use an ORT 1.24+ DirectML-enabled runtime and a usable adapter for the DirectML example; a CPU-only runtime cannot
satisfy that target. CPU uses `kind: builtin`, `ep_name: CPUExecutionProvider`, and no provider options; registration
is implicit but assignment/output checks still apply. DirectML uses `kind: builtin`, `ep_name: DmlExecutionProvider`,
and string option `device_id: "0"`. Its internal adapter loads the public
`OrtSessionOptionsAppendExecutionProvider_DML` export for the ORT 1.24 path and disables memory patterns.

CUDA uses `ep_name: CUDAExecutionProvider` and an internal `AppendExecutionProvider_CUDA_V2` adapter. Other supported
built-ins use generic `AppendExecutionProvider(ep_name, options)` with the full canonical name, not a profile alias.
ORT 1.24.4 CPU execution and generic canonical `AzureExecutionProvider` append were checked against the actual runtime.
The latter reached the expected CPU-fallback prohibition, not successful Azure execution. DirectML's public append
entry point was resolved, but device setup failed on this host; CUDA execution remains unchecked. Availability, API
compatibility, and successful device execution are separate checks. An unavailable adapter/provider fails instead of
selecting CPU.

## Run a plugin

Supply a profile with the following implemented fields, substituting a compatible plugin library and its actual
factory-reported name. This is a template, not an assertion of a checked-in plugin profile or validated WebGPU run:

```json
{
  "schema_version": 1,
  "kind": "plugin",
  "ep_name": "ActualReportedExecutionProvider",
  "library": "./provider.dll",
  "registration_name": "ep_op_conformance_plugin",
  "device_index": 0,
  "options": {},
  "skips": {}
}
```

`library` is relative to the profile directory unless absolute. `registration_name` identifies the library to ORT;
`ep_name` selects matching devices returned by `GetEpDevices`, and `device_index` selects within that matching list
(default zero). The runner appends the chosen device using `AppendExecutionProvider_V2`. No matching device fails
the run; it does not cause a CPU substitution or mass skip.

```powershell
& $Runner --runtime C:/sdk/ort/onnxruntime.dll --profile C:/ep/plugin-profile.json `
  --cases "$DataCases" --report "$RunRoot/plugin.json"
```

A profile may use `skips` as an exact selected-case-ID to nonempty-reason map. Unknown skip IDs fail input validation.
Use the same model/data/assertion artifacts for each target, with skips/options only in its profile. Partial support
must not silently use CPU. A sample plugin's supported Mul case and unsupported Add case are useful harness probes,
but sample-plugin execution is not WebGPU validation. Newly skipping required WebGPU behavior cannot satisfy the
future extraction gate.

## Export the C++ fixture

The exporter is ORT-private producer code, not part of the standalone consumer dependency graph. It is enabled per
tester by `ConfigEpOpConformanceExport(directory, id)`, not by the environment variable alone. Capture occurs once per
configured tester before the provider loop, then original inference and follow-up assertions continue. A second run
of that same tester does not capture a second invocation. No existing shared authored variants have been retired.

Only the positive Mul fixture honors `ORT_EP_OP_CONFORMANCE_EXPORT_DIR` for persistent output; the other fixture tests
use temporary directories. The exporter refuses to overwrite an existing case directory. Unsupported custom
verifiers, expected failures, disabled/sorted verification, custom registries/session/run/resolve settings, and
unsupported value/graph forms are rejected rather than silently approximated. Multiple-run configurations are also
rejected. Follow-up C++ assertions and WebGPU applicability need source audit and coverage evidence; exporting bytes
alone does not prove migration or parity.

For an already configured main ORT source build with unit-test support, use its directory containing `CMakeCache.txt`:

```powershell
$OrtBuild = 'C:/build/ort/Release'
cmake --build "$OrtBuild" --config Release --target onnxruntime_ep_op_conformance_export_test example_plugin_ep
$CppCases = Join-Path $RunRoot 'cpp_cases'
$env:ORT_EP_OP_CONFORMANCE_EXPORT_DIR = $CppCases
try {
  & "$OrtBuild/Release/onnxruntime_ep_op_conformance_export_test.exe" '--gtest_filter=EpOpConformanceExportTest.*'
} finally {
  Remove-Item Env:ORT_EP_OP_CONFORMANCE_EXPORT_DIR -ErrorAction SilentlyContinue
}
& $Runner --runtime C:/sdk/ort/onnxruntime.dll `
  --profile "$Repo/onnxruntime/test/ep_op_conformance/examples/cpu.json" `
  --cases "$CppCases" --report "$RunRoot/cpu-exported.json"
```

Adjust the executable path to the main build's generator/layout. The `example_plugin_ep` build target is a separate
harness fixture, not the exporter or a WebGPU plugin. Use a fresh `$CppCases` root for each persistent export.

## Migration and rollout

The [migration workflow](../../../docs/design/webgpu_ep_extraction/test_ownership_and_conformance/ep_operator_conformance_design.md#c-migration-workflow-proposed)
is documentation-only in this revision: export concrete models/inputs/authored expectations/comparison policies,
convert artifacts to readable JSON for check-in, then regenerate artifacts for the same runner. Preserve graph,
dtypes, initializer roles, assignment, and assertions; reject unrepresentable conversion. Validate artifact-to-JSON-
to-artifact round trips, equivalent execution, and C++ assertions outside export. Retire the corresponding C++ only
when all applicable existing EP consumers retain equivalent coverage. JSON then becomes the single canonical source;
no permanently dual-maintained corpus, and generated binaries are always derived. No converter or existing case
migration is delivered here.

Start with CPU CI, then require all supported built-in EPs in applicable CI lanes against the same corpus with
explicit, reviewable support exclusions. Plugins use the same machinery. Incrementally migrate existing generic
operator C++ tests, adding tester capabilities only for concrete needs with focused tests. The population is the
repository's generic operator tests, including additions/changes during migration; a pinned snapshot or ledger is
optional tracking, not the completion definition.

Before extraction, **all WebGPU-applicable generic tests** need equivalent execution and relevant assertions in
applicable WebGPU lanes. Export alone or skips because the tester lacks functionality do not count. Broader generic
migration continues after this milestone; EP-specific tests and browser/Wasm continuity remain separate.

Broader dtype support is a follow-up: use per-tensor types, preserve integer precision and initializer roles, and
define comparison by dtype. Do not implement mixed-type inputs/outputs by casting through float32 storage.

## Integration checks

The producer-side [tools/test_pilot.py](tools/test_pilot.py) checks CPU execution, wrong outputs/shapes, malformed
artifacts, unsupported fields/profiles, deterministic generation, unavailable providers/devices, and non-green
empty/all-skipped runs.
With the matching sample-plugin library it also checks Mul execution and assignment, initializer handling, wrong
expected output, and unsupported Add rejection with CPU fallback prohibited. With captured C++ cases it replays
those exact artifacts on both CPU and the sample plugin.

```powershell
python onnxruntime/test/ep_op_conformance/tools/test_pilot.py `
  --runner "$Runner" --runtime "$OrtBuild/Release/onnxruntime.dll" `
  --plugin-library "$OrtBuild/Release/example_plugin_ep.dll" --exported-cases "$CppCases" -v
```

Use a runtime and sample plugin built from matching sources; the current example plugin requires newer APIs than
the runner's minimum. Omitting an optional plugin/exported-cases argument reports those tests as skipped, not
validated. To register this suite in the standalone CTest build, reconfigure it with
`EP_OP_CONFORMANCE_TEST_RUNTIME`, `EP_OP_CONFORMANCE_TEST_PLUGIN`, `EP_OP_CONFORMANCE_TEST_CPP_CASES`, and `Python3_EXECUTABLE` pointing
to these same files/directories and the producer virtual environment. Then run `ctest --test-dir "$PilotBuild" -C
Release --output-on-failure`. None of these developer-test settings introduces Python into native runner execution.

The checked-in sample profile resolves `example_plugin_ep.dll` beside the profile. Stage both together or use a
profile with an absolute library path. Its reasoned Add skip applies to the three data-defined cases, not a
single exported-Mul selection: skip IDs must be present in the selected corpus.

## CPU CI and staged execution

[Windows CPU source CI](../../../.github/workflows/windows_x64_release_build_x64_release.yml) enables the main-build
runner and generates canonical cases once. It publishes separate CPU and infrastructure build handoffs from its
existing build; no additional full ORT build is introduced. Two independent reusable workflows consume immutable IDs:

- [Provider conformance](../../../.github/workflows/windows-ep-op-conformance.yml), called with
  `ep-name: CPUExecutionProvider`, downloads only the main-build runner/runtime,
  canonical cases, CPU profile, and generation metadata. It invokes the native runner directly and retains its report,
  without checkout, Python setup, a plugin fixture, or infrastructure tests.
- [Infrastructure validation](../../../.github/workflows/windows-ep-op-conformance-infrastructure.yml) checks out the same
  revision to run generator/package tests and C++ export/replay integration checks using relocated build outputs.
  It publishes hash-manifested source-kit, case/scenario, and matching runtime/plugin artifacts. A no-checkout job
  verifies them and compiles packaged source with kit-local public headers and pinned public dependencies. A fresh
  execution job has neither source/build trees nor Python setup, restricts PATH to System32, and runs 20
  matching-runtime CPU/plugin checks plus nine separate ORT 1.24.4 CPU compatibility checks.

CPU provider conformance does not depend on infrastructure test or source-kit build success. The current plugin is
never forced onto the minimum runtime. The CPU caller and all infrastructure jobs explicitly select the
`onnxruntime-github-vs2022-latest` 1ES pool with distinct run-specific job IDs. The infrastructure workflow fixes its
pool internally rather than exposing a runner input. The Windows host still supplies PowerShell, system libraries, and the MSVC
redistributable. Reports, provenance, and build diagnostics are retained on failure. Required-check owners must review
the separate CPU and infrastructure check names. The revised workflow DAG has not run on GitHub. Ordinary EP lanes
retain their normal environments.

### Reusing provider execution

The reusable [provider workflow](../../../.github/workflows/windows-ep-op-conformance.yml) selects the EP through the
bundle's `profile.json`, not through workflow logic. `ep-name` uses the actual EP name, such as `CPUExecutionProvider`,
in job/report labels; it does not override the profile's EP registration name. Call it with an immutable `bundle-id`
and a required JSON array of `runs-on` labels; there is no default runner:

```yaml
  ep_op_conformance_cpu:
    needs: build_x64_release
    uses: ./.github/workflows/windows-ep-op-conformance.yml
    permissions:
      contents: read
    with:
      bundle-id: ${{ needs.build_x64_release.outputs.conformance-cpu-id }}
      ep-name: CPUExecutionProvider
      runs-on: >-
        ["self-hosted",
         "1ES.Pool=onnxruntime-github-vs2022-latest",
         "JobId=ep-op-conformance-cpu-${{ github.run_id }}-${{ github.run_number }}-${{ github.run_attempt }}"]
```

The Windows bundle contains `bin/onnxruntime_ep_op_conformance_test.exe`, `bin/onnxruntime.dll`, `cases/`,
`profile.json`, and optional `generation.json`. Include provider DLLs and native dependencies as required. Plugin
library paths are resolved relative to the profile; keep them bundle-relative. Callers supply a compatible runtime,
profile/options/skips, and Windows runner labels with the required hardware, drivers, and host dependencies already
available. Report artifact names include `ep-name`; repeated invocations for the same EP in one run currently share
an artifact name and must be avoided.
The workflow performs no EP-specific installation, and adding a caller does not add a new runner adapter or establish
hardware coverage. Linux/macOS bundle layouts and browser/Wasm execution are not provided by this Windows workflow.

### Main-build CPU execution

Use the normal ORT build/test Python environment with NumPy and ONNX installed, then enable the opt-in main CMake
switch. Generation uses the build's `Python_EXECUTABLE`:

```powershell
python tools/ci_build/build.py --config Release --build_dir build/Windows --build_shared_lib --update --build --parallel `
  --cmake_extra_defines onnxruntime_BUILD_EP_OP_CONFORMANCE_TESTS=ON
$MainBuild = 'build/Windows/Release'
$Generated = "$MainBuild/ep_op_conformance/generated"
$Generation = Get-Content "$Generated/current.json" -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force "$MainBuild/ep_op_conformance/reports/Release" | Out-Null
& "$MainBuild/ep_op_conformance/Release/onnxruntime_ep_op_conformance_test.exe" `
  --runtime "$MainBuild/Release/onnxruntime.dll" `
  --profile onnxruntime/test/ep_op_conformance/examples/cpu.json `
  --cases "$Generated/$($Generation.directory)" --report "$MainBuild/ep_op_conformance/reports/Release/cpu.json"
if ($LASTEXITCODE -ne 0) { throw 'CPU EP conformance failed' }
```

The direct target `onnxruntime_ep_op_conformance_test` also builds its runtime and case prerequisites. C++20 remains
target-local. Static-only, cross-compiled, browser, minimal, and reduced-op builds fail explicitly when opted in.
[tools/build_cases.py](tools/build_cases.py) records input/producer hashes in digest-keyed immutable build directories;
installed NumPy, ONNX, and protobuf versions are recorded, not separately pinned or enforced. Changed inputs or
producer versions select a new directory and unchanged builds avoid regeneration. Building the runner does not
register provider execution with CTest; CI or developers invoke it explicitly with a runtime, profile, cases, and
writable report path. Comparator/exporter tests retain CTest integration. Generation metadata lives under the build's
`ep_op_conformance/generated`; the example above writes reports under `ep_op_conformance/reports/<configuration>`.

### Source-kit preparation

[tools/package_ci.py](tools/package_ci.py) provides `create`, `verify`, and `stage` commands. `create` takes
`--generated`, `--runner`, `--runtime`, `--plugin`, `--revision`, and a fresh `--output`. It packages source with public
header closure/notices/contracts, exact main-build positives plus derived scenarios, and matching native fixtures
as three independent artifacts. `verify` takes `--source`, `--cases`, `--fixture`, and optional `--revision`.
`stage` adds `--runner` and a fresh `--output`; optional `--minimum-runtime` selects CPU-only smoke instead of the
matching fixture. Scenario bytes/profiles are checked against the producer package, never maintained as goldens.
[tools/test_infrastructure.py](tools/test_infrastructure.py) checks generation reuse/invalidation, immutability,
package hashes, revision agreement, and header closure using the ORT build/test Python environment.

To rehearse with that environment and a fresh destination:

```powershell
python onnxruntime/test/ep_op_conformance/tools/prepare_ci.py `
  --runner "$Runner" --runtime C:/sdk/ort-1.24.4/onnxruntime.dll --output "$RunRoot/cpu-bundle"
& "$RunRoot/cpu-bundle/run_ci.ps1" -ExpectedRuntimeVersion '1.24.4'
```

Preparation writes source/public-header/tool hashes, runtime and bundle-file hashes, and producer package versions
to `prepared.json`; packaged staging also retains all three manifests and generation metadata. Consumption checks
file hashes before execution and retains this provenance with reports. This is bounded bundle provenance, not
complete device/plugin provenance or a released conformance kit. Outputs must be fresh and are never overwritten.

| Stage | Required dependencies |
| --- | --- |
| Producer/build | Checkout, C++20/CMake, public ORT/JSON headers, matching ONNX/protobuf headers/libraries; Python/NumPy/ONNX for JSON generation and fixture preparation; ORT-private build only for optional C++ capture |
| Native execution | Runner and native dependencies, compatible ORT runtime/provider dependencies, generated case artifacts, target profile, optional plugin; writable report directory |

The staged executor must not invoke the generator, Python tests, or a private ORT test binary. Prepare negative
fixtures in advance; invoke the same runner for each and check the expected exit status/report diagnostic, not
merely any failure. Native dependency/runtime load failure must not accidentally satisfy an output-mismatch check.
Preserve reports as CI artifacts. CPU is only the first integration; other built-in applicable lanes and plugin
consumers use the same corpus and checks with target-specific setup/options and reviewable support exclusions.

## Reports and validation status

The runner prints and writes JSON with schema version, loaded runtime version, profile, per-case status, duration,
assignment evidence on successful assignment, and failure phase/reason where available. Cases are `PASS`, `FAIL`, or
profile-declared `SKIP`. Session, assignment, and execution/comparison errors fail the case; setup/input errors give
run status `ERROR`. Exit code is 0 for a run with at least one pass and no failures, 1 for case failure or no passes
(including all skipped), and 2 for setup/input/report errors. Empty selection, duplicate IDs, or invalid data do not
pass. A profile skip does not bypass artifact validation. A runner PASS covers only selected cases, not complete
generic-test population or WebGPU parity accounting.

Producer `source` metadata records the JSON definition filename or C++ suite/test/file/line. For JSON generation,
the top-level `id` identifies the authored entry without repeating it in `source`. Each case result retains that
metadata; the report also retains the loaded runtime version and target profile. Full generic report hashes, device/environment
provenance, and population reconciliation remain future work. Keep artifact source metadata; no original source
files are required at execution. See the [report contract](../../../docs/design/webgpu_ep_extraction/test_ownership_and_conformance/ep_operator_conformance_design.md#runner-inputs-and-reports).

Revised local Windows validation:

- The explicit CPU job invocation passes all three canonical cases with assignment/output checks; CPU execution is
  absent from CTest. Direct-target changed-input and unchanged-input rebuilds pass; static-only, minimal, and
  reduced-op opt-ins reject explicitly. Twenty-eight
  generation/package tests pass. The final source kit built outside the checkout with public dependencies; after
  removing its source/build trees, all 20 matching-runtime CPU/plugin and nine ORT 1.24.4 CPU scenarios passed.
- C++20 builds and 23 exporter tests, 36 integration checks, and 10 comparator checks passed. All 36 integration
  checks ran with the matching source-built runtime/sample plugin and fresh single-run C++ artifacts; none skipped.
  Captured and data-defined artifacts replayed on CPU and the sample plugin, including staged native execution.
- The ORT 1.24.4 CPU runtime passed the applicable integration suite; seven optional plugin/C++ replay tests are
  skipped when those arguments are omitted. A fresh bundle built with CI-pinned dependencies passed all nine native
  scenarios without Python or source tools on PATH. This is local rehearsal, not a remotely passed workflow.
- Canonical generic Azure append on ORT 1.24.4 reached the expected session failure with CPU fallback disabled;
  no Azure inference pass is claimed. CUDA execution remains unchecked.
- DirectML on ORT 1.24.4 failed setup because no usable display adapter was available; it did not pass via CPU.

Real WebGPU and an existing non-plugin non-CPU EP still need successful runs. The sample plugin validates the
harness, not WebGPU, and assignment proves ORT dispatch rather than internal hardware execution. No existing
generic C++ cases have been migrated/retired. WebGPU generic parity with additions/changes accounted for, equivalent
coverage for all existing consumers before C++ retirement, and separately owned EP-specific/browser continuity are
still required before extraction. The pilot subset, successful builds, or converter alone cannot satisfy that gate.
