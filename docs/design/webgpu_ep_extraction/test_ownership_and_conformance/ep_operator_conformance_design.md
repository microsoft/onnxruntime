# Execution Provider Operator Conformance Suite

Status: Revised design addressing the agreed pilot review and CI organization. The bounded implementation passes local
checks; isolated CPU CI is implemented and locally rehearsed, but has not run on GitHub. Source-kit packaging,
main-build integration, and EP workflow rollout remain planned. Supports
[Test Ownership and Operator Conformance](test_ownership_and_conformance_workstream.md). Local commands are in the
[pilot README](../../../../onnxruntime/test/ep_op_conformance/README.md).

## Purpose

Define a reusable ONNX Runtime (ORT) operator conformance suite that can validate in-tree and external execution
providers (EPs) against common operator semantics.

The short name is **EP op conformance**. The broader term **EP conformance** is reserved for tests of other provider
contracts, such as registration, device discovery, and lifecycle. This suite checks assignment and fallback only to
establish which EP executes the operator cases; its infrastructure regression checks do not broaden the suite's scope.

The primary external audience is EP implementers. They should be able to build a standalone runner from a source kit
and test their plugin without cloning or building the ORT repository or linking private ORT libraries. Native
execution must not require Python or the source kit. The suite should also preserve the value of ORT's existing
operator tests while separating portable test cases from ORT-private C++ test infrastructure.

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

## Scope and decisions

- Establish one shared tester, then integrate **all supported built-in EPs** in their applicable CI lanes using the
  same canonical corpus and explicit, reviewable support exclusions. CPU CI is the starting integration, not the
  adoption target. Plugin EPs use the same cases, execution, and comparison through plugin setup.
- Incrementally migrate the repository's existing **generic operator C++ tests**. Add tester functionality only as
  concrete migration cases require it, with focused tests for each extension. Do not require every capability upfront.
- The population is the repository's generic operator tests, including additions and changes during migration.
- Before WebGPU extraction, migrate **all WebGPU-applicable generic operator tests**, preserving equivalent execution
  and relevant assertions in applicable WebGPU lanes. Export alone or a skip for missing tester functionality is not
  migration. Broader repository migration continues after that milestone.
- Use public ORT APIs at the consumer boundary and ORT 1.24 as the minimum for graph-assignment APIs. Check actual
  runtime/adapter compatibility; this minimum does not guarantee operator, opset, device, or provider support.
- Deliver native dynamic runtime/plugin execution first. Browser/Wasm coverage and EP-specific test continuity have
  separate owners and gates; a native pass does not replace them. A native static host/SDK is deferred.
- Maintain one canonical authored definition per case and shared validators. Generated binaries are derived outputs,
  never a second maintained corpus. External execution must not require an ORT checkout/build or Python.

EP-specific unit-test migration belongs to provider code migration. This work does not expose `OpTester`,
`ModelTester`, or private ORT libraries as a public ABI, require every EP to implement every case, replace all
provider regression tests, or define operator semantics independently from ONNX.

## Proposed components

The design separates case authorship, case distribution, and execution:

1. **Canonical source** for new portable cases is readable JSON. An existing C++ variant remains canonical until its
  migration and all-consumer retirement gate are satisfied.
2. **Producers** are the JSON-to-artifact generator and the ORT-private C++ exporter. Both produce models, tensors,
  and manifests using the same artifact contract. A proposed artifact-to-JSON converter supports migration from
  exported C++ artifacts to JSON authorship.
3. **Shared native consumer** loads those artifacts, configures the target through a public setup adapter, checks
  assignment, executes the session protocol, compares outputs, and writes a report.
4. **Provider profile** supplies target setup, options, and explicit support expectations, not alternate models,
  inputs, expected values, or validator implementations.
5. **EP op conformance kit** will supply versioned standalone runner source, immutable generated cases, contracts, and
  provenance for external use. Prebuilt runners are optional conveniences for tested platforms.

The portable contract is the artifact format and execution/result semantics. The chosen initial consumer is a
standalone native C++20 executable using public ORT headers, ONNX protobuf, and nlohmann JSON. It loads the ORT runtime
dynamically and does not link private ORT test libraries. Python is producer/test tooling only, not a runner runtime
dependency.

## Relationship to `onnxruntime_provider_test`

`onnxruntime_provider_test` should remain the comprehensive in-tree provider regression executable. It can use
ORT-private helpers and cover implementation details that are inappropriate for an external contract.

The EP op conformance suite should be a second layer:

| Layer | Purpose | Dependencies |
| --- | --- | --- |
| `onnxruntime_provider_test` | In-tree provider and operator regression testing | ORT-private libraries and helpers |
| `onnxruntime_ep_op_conformance_test` | Portable EP operator conformance | Public ORT APIs and released case data |

## Case ownership and storage

ORT owns canonical portable cases, producer tools, shared execution/validation code, and migration verification.
The current pilot lives under `onnxruntime/test/ep_op_conformance/`, with `cases/pilot.json`, `tools/create_cases_from_json.py`,
`runner/`, and `examples/`. The private producer is in `unittest_util/ep_op_conformance_export.cc` and `base_tester`.
The versioned draft contracts below describe the bounded pilot; they are not formal JSON Schema implementations.
Conversion tooling, broader schemas, and release-kit packaging remain future work.

Standard ONNX cases should reuse or derive from ONNX backend test data where practical. ORT owns cases for ORT contrib
operators and generic ORT EP behavior.

Generated models, tensors, and manifests are immutable derived outputs, never a second hand-maintained golden corpus.
Producer CI should freeze concrete bytes and generation seeds and verify repeatability. Check in exact serialized
models only when serialization itself matters or deterministic generation is impractical. Keep authored expected
values; a CPU or plugin fetch must not silently replace them.

Provider repositories own their profiles and implementation-specific tests. For example:

```text
webgpu-ep/tests/ep_op_conformance/
  provider-profile.json
```

## Case representation and authorship

The authoring JSON is readable source; the generated manifest is an execution index, not another authored test.

```text
existing C++ -> opt-in exporter -> concrete artifacts -> proposed converter -> readable JSON
new portable cases -------------------------------------------------------> readable JSON

readable JSON (canonical at transfer)
  -> JSON-to-artifact generator
  -> model.onnx + tensors + case.json
  -> shared native runner + target profile
  -> JSON report + exit status
```

The exporter can already supply the runner directly for transition/differential checks. The converter and migration
of existing cases are proposals, not requested implementation in this pilot revision. The new-JSON path also goes
through the generator, not through the converter.

The model owns graph structure, attributes, opsets, graph types/shapes, and initializer roles. Named ONNX
`TensorProto` files own concrete input/expected-output payloads. The manifest owns identity, paths, comparisons,
assignment requirements, and source provenance. Do not duplicate values in the generated manifest. The generator
uses `test_data_set_0/`; the exporter uses flat tensor files. Manifest-relative paths support either layout.

### Authoring JSON v1 (revised draft)

The bounded generator represents one standard-domain node without attributes, subgraphs, or functions. Its model
uses IR version 10, one authored opset import, and node name `node1`. This is not yet a general ONNX graph language.
Unknown fields and unsupported representations must be rejected, not silently ignored.

| Object | Fields and constraints |
| --- | --- |
| Document | Required `schema_version: 1`; nonempty `cases` array |
| Case | Required `id`, `operator`, `opset`, `inputs`, `outputs`, `comparison`; no run-count field |
| `id` | Structured case ID following the convention below; unique ignoring ASCII case; stable semantic identity, not an ordinal |
| `operator`, `opset` | Standard ONNX operator name and supported positive integer opset; generated model must pass ONNX checking |
| `inputs` | Ordered array of tensor definitions; order is the node input order; unique nonempty names |
| `outputs` | Nonempty ordered array of authored expected tensors covering all outputs; unique names |
| Tensor | Required `name`, `shape`, `values`; optional boolean `initializer` for an input, default false |
| `shape`, `values` | Concrete nonnegative integer dimensions; `[]` means scalar; float32 values with exactly the shape's element count; zero dimensions imply zero elements |
| `initializer` | True retains that input in the model's initializers and omits it from runtime feeds; not a comparison option |
| `comparison` | Required `atol`, `rtol` (finite nonnegative numbers) and `nan_equal` (boolean); expanded to a policy for each output |

New IDs follow the [case ID convention](../../../../onnxruntime/test/ep_op_conformance/README.md#case-id-convention):
`<domain>.<operator>.opset<version>.<descriptive_variant>`, with no mandatory dtype slot. IDs are stable and
provider-neutral, while the model and tensors remain authoritative. Dots separate components, and the final variant
contains only ASCII letters, digits, and underscores. The generator validates the structured syntax and
case-insensitive uniqueness, then writes `case-<id>` directories. This prefix avoids Windows reserved device names
without restricting domain prefixes. Manifests, reports, and skip profiles retain the unprefixed logical ID.

This subset has implicit float32 type, finite JSON values representable as finite float32, one generated dataset, and
one execution per dataset. No `repeat_count` exists in authoring or manifest v1. JSON definitions cannot express
missing optional slots, nonstandard domains, attributes, arbitrary graphs, expected errors, or custom validators.
The generator must reject these rather than coerce them into this subset. Authored values are converted to float32
once; generation must not obtain expected values by executing the EP under test.

```json
{
  "schema_version": 1,
  "cases": [{
    "id": "ai.onnx.Mul.opset13.float32_scalar",
    "operator": "Mul",
    "opset": 13,
    "inputs": [
      {"name": "X", "shape": [], "values": [2]},
      {"name": "Y", "shape": [], "values": [3], "initializer": true}
    ],
    "outputs": [{"name": "Z", "shape": [], "values": [6]}],
    "comparison": {"atol": 0, "rtol": 0, "nan_equal": false}
  }]
}
```

### Generated manifest v1 (revised draft)

| Field | Meaning and constraints |
| --- | --- |
| `schema_version` | Required integer `1`; unknown versions fail |
| `id` | Required nonempty stable ID; duplicate IDs within a selection fail |
| `model` | Required relative path to the serialized ONNX model |
| `required_nodes` | Nonempty array of unique nonempty node names present in the model; all must be assigned to the selected target |
| `datasets` | Nonempty array; each object has `inputs` and `outputs` arrays of relative TensorProto paths |
| `comparison` | Output-name to `{atol, rtol, nan_equal}` map; one policy for every expected model output, no extra entries |
| `source` | Producer provenance object; retain it with artifacts, not as a runtime dependency on the original source |

Each dataset supplies named feeds and every authored expected output exactly once. Initializers remain in the
model, not the feed list. Tensor names, not filenames or list positions alone, identify model inputs/outputs.
Paths must stay inside the case directory, including after symlink resolution; absolute paths and traversal fail.
Missing files/datasets, duplicate tensor names, malformed tensors, unknown fields, and unsupported versions fail
closed. There is no execution-count field or implicit repeated-run protocol.

```json
{
  "schema_version": 1,
  "id": "ai.onnx.Mul.opset13.float32_scalar",
  "model": "model.onnx",
  "required_nodes": ["node1"],
  "datasets": [{"inputs": ["test_data_set_0/input_0.pb"], "outputs": ["test_data_set_0/output_0.pb"]}],
  "comparison": {"Z": {"atol": 0, "rtol": 0, "nan_equal": false}},
  "source": {"definition": "example.json"}
}
```

The pilot tensor encoding is named, inline, dense float32 only: concrete dimensions, checked element/byte counts,
either `float_data` or correctly sized `raw_data`, no external data or segments. Raw float bytes require a
little-endian host in this pilot. Sparse tensors, containers, optional values, strings, other dtypes, and mixed-type
storage are not implemented. Export additionally rejects functions/training graphs, nested/sparse graph attributes,
nonstandard-domain nodes, and unsupported session/assertion behavior. Producer checks and runtime validation must
not silently remove unsupported semantics; the exporter and JSON-to-artifact generator need not have identical
authoring expressivity.

### Temporary C++ canonical source

The pilot is explicit opt-in through `ConfigEpOpConformanceExport(directory, id)`. `BaseTester::RunWithConfig` captures
once per configured tester, after model construction and feed preparation but before the provider loop and its
assignment mutations. It captures authored expected data, then continues original inference and assertions. It is
capture-and-continue, not export-and-return or C++ source translation. Reusing that tester does not export a second
invocation; separate variants need explicit identities and capture instances.

The exporter rejects `num_run_calls_ != 1`: multiple-run C++ cases are not approximated by a single run. It also
rejects unsupported custom verifiers, disabled verification, expected failures, custom registries, custom
session/run/resolve settings, sorted outputs, missing optional values, and unsupported tensor/graph forms.
It refuses to overwrite an existing case directory. These checks prevent silent weakening of supported exports;
they do not discover arbitrary assertions after `Run()`/`GetFetches()`, provider-dependent guards, or applicability.
Those require source audit and equivalence checks before claiming the original test's coverage is preserved.

### C++ migration workflow (proposed)

The target set of generic C++ tests defines migration scope, including parameter/loop variants and additions or
changes during migration. Preserved coverage means equivalent execution and assertions to the original C++ test for
every applicable existing EP consumer and CI lane. Migration moves the test to the canonical conformance form and
removes the original C++ case. It requires verified equivalent coverage in blocking CI for every applicable existing
EP consumer and lane. A retained C++ copy, including a placeholder, means the test is not fully migrated; passing a
selected subset of portable cases is not sufficient.

1. Identify the concrete generic test variant, applicable EP consumers/lanes, and every protected assertion. Export
  the unassigned model, named inputs, authored expected outputs, explicit comparison policies, required nodes, and
  provenance. Run original checks; never substitute observed CPU/plugin outputs for authored expectations.
2. Convert those artifacts into readable JSON for review/check-in. Preserve stable identity, graph semantics,
  opsets/attributes, graph types/shapes, concrete payloads, initializer-versus-feed roles, assignment requirements,
  and comparison/assertion meaning. Reject any form the converter or JSON contract cannot faithfully represent.
  The current single-node JSON subset cannot represent every exported graph.
3. Add only the missing tester/encoding/validator capability demanded by that case, with focused positive and
  negative tests. Unsupported export/conversion is a migration blocker, not a support exclusion for parity.
4. Regenerate artifacts from the candidate JSON and validate the **artifact -> JSON -> artifact round trip**.
  Compare typed payloads, graph semantics, initializer roles, assignment and comparison policies; verify repeatable
  generation and exact bytes where serialization is part of the test. Explain harmless serialization differences
  instead of weakening semantic checks. Execute regenerated artifacts using the same runner and applicable EP lanes.
5. Audit C++ assertions outside export, including custom callbacks, `GetFetches()` assertions, inferred shape checks,
  guards, provider-dependent tolerances/options, errors, and lifecycle behavior. Prove equivalent checks/execution,
  not merely equivalent output bytes. Classify tests by behavior: portable operator tests belong in conformance;
  provider-specific and host-specific tests remain in their respective suites. Split mixed tests where useful.
6. Transfer canonical ownership to JSON and remove the original C++ case **only after every applicable existing EP
  consumer executes the equivalent replacement in blocking CI in its applicable lanes**. Until then, keep C++
  canonical and preserve its execution. An artifact consumer may reuse shared validators, but must not retain a copy
  or placeholder of the original C++ case. Keep unrelated variants intact.

Temporary differential execution is useful; permanently maintaining C++ and JSON definitions of the same case is
not. Generated ONNX/tensor/manifest files remain immutable derived outputs. This revision documents the converter
and migration acceptance criteria; it does not implement conversion or migrate existing repository cases.

## Execution semantics

For an ordinary operator conformance case, the runner should:

1. Load a compatible ORT runtime and configure the requested target through its public setup adapter.
2. Apply the requested EP options.
3. Create a session with CPU fallback disabled through `session.disable_cpu_ep_fallback`, unless the CPU EP is
   itself the target.
4. Load the generated or packaged model.
5. Enable `session.record_ep_graph_assignment_info` and use public assignment APIs to require the named nodes on the
  named target EP. Default to `ORT_DISABLE_ALL`; required transforms can still occur, so missing nodes cannot pass.
6. Execute once per input dataset, in order, on that case's session. There is no repeat loop or run-count option.
7. Compare every output according to the case's comparison policy.
8. Record diagnostics without changing the defined outcome.

Disabling CPU fallback is essential. A correct result produced by the CPU EP does not demonstrate conformance of the
target EP. The exception is the CPU reference run, where the CPU EP is the target: there is nothing to fall back
from, and ORT rejects a session that disables CPU fallback while nodes are assigned to the CPU EP.

Public assignment evidence plus successful execution establishes dispatch under this contract, not the EP's internal
hardware execution. The pilot rejects every reported assignment to another EP. Future cases intentionally requiring
partial assignment need explicit node-scoped contracts, separately classified from strict cases; do not implicitly
allow CPU helpers for an entire provider.

Target setup is the only provider-dependent runner boundary:

| Target | Public setup | Fallback and assignment |
| --- | --- | --- |
| Built-in CPU | Implicit CPU registration on an ordinary session | Do not disable CPU fallback; require `CPUExecutionProvider` |
| Other built-in EP | Internal provider-specific public adapter, or generic append with canonical `ep_name` where supported | Forbid CPU fallback and require the canonical target EP name |
| Plugin EP | Register library, `GetEpDevices`, select device, `AppendExecutionProvider_V2` | Forbid CPU fallback and require the reported target EP name |

Do not register legacy provider-bridge DLLs as plugin API libraries or assume all legacy EPs support `GetEpDevices` or
generic string append. A missing requested adapter/provider/device fails explicitly, never substitutes CPU. Compatible
runtime/provider dependencies and platform-specific adapter builds remain necessary; one corpus does not imply one
universal runtime package.

## Result semantics

Each selected case should produce exactly one result:

- `PASS`: The target EP executed the required graph and all outputs matched.
- `FAIL`: Output mismatch, or the target EP did not execute the required graph.
- `SKIP`: The provider profile says the case is not expected to run, with a documented reason.

A skip is legitimate only when the provider profile says so. If the profile does not skip a case and the target EP
will not run it, that is a failure.

A skip should identify a stable case ID, a reason, and preferably a tracking issue. Broad wildcard skip lists
should be discouraged because they obscure coverage loss.

If no compatible device is available, the run fails rather than reporting every case as skipped.

Invalid versions/profiles, corrupt cases, unknown skip IDs, absent datasets, and empty selection must fail closed.
The pilot records run-level setup/input errors separately and returns nonzero for them; per-case session, assignment,
or output errors are failures. An all-skipped run is not a successful run. Migration/parity acceptance must additionally
verify equivalent execution and assertions for every C++ test in the target set, not trust a selected-case pass count.

## Comparison semantics

The case owns the default comparison policy. The policy may specify:

- Exact comparison.
- Absolute and relative tolerances.
- NaN and infinity handling.
- Type-specific rules.
- Ordering rules where the operator permits multiple valid orders.

A provider may eventually define a narrow override when implementation precision requires it. Every override must
include a reason, be visible in the report, and not waive the migration/parity gate without review. Provider-wide
tolerance inflation is not acceptable. Profile comparison overrides are not implemented in the pilot.

The shared comparator requires an actual dense float32 output with exactly the expected rank, dimensions, and element
count, without broadcasting or dtype conversion. Finite float32 values are compared using double-precision arithmetic:

$$
|a-e| \leq \mathrm{atol} + \mathrm{rtol}\,|e|
$$

Here $a$ is actual and $e$ is authored expected, with finite nonnegative tolerances. Both tolerances zero gives exact
numeric equality, not byte equality; positive and negative zero match. NaNs match only when both are NaN and
`nan_equal` is true; a lone NaN always fails. Infinities match only when both are infinite with the same sign;
finite/infinite pairs fail regardless of tolerance.
Use this positive acceptance condition so a NaN cannot accidentally pass an inverted mismatch check. Other types,
correlated-output rules, sorting, signed-zero-specific checks, and custom portable validators remain future work.
Broader dtype support must use per-tensor types and dtype-specific comparison, preserving integer precision and
initializer roles. Mixed inputs/outputs must not pass through float32 storage. Add it when concrete migrations need it.

## Target profile v1 (revised draft)

A profile selects setup and declares reviewable support exclusions; it never supplies alternate models, expected
values, or validators. It must not become a second `GetCapability()` implementation. Unknown fields fail.

| Field | Meaning and constraints |
| --- | --- |
| `schema_version` | Required integer `1` |
| `kind` | Required `builtin` or `plugin` only; CPU is `builtin`, not a third kind |
| `ep_name` | Required canonical full EP name for availability, assignment, and generic append; the only provider identity field |
| `options` | Optional string-to-string EP options map, default empty; CPU accepts no options |
| `skips` | Optional exact selected-case-ID to nonempty reason map, default empty; unknown IDs fail; include a tracking issue where practical |
| `library` | Plugin-only required library path; relative to the profile directory unless absolute |
| `registration_name` | Plugin-only required library registration name, distinct from the EP's canonical identity |
| `device_index` | Plugin-only nonnegative integer, default zero, indexing devices matching `ep_name` |

`provider_name` is not a supported field. Built-in profiles cannot contain plugin fields. CPU uses implicit
registration while retaining output and assignment checks. Built-in availability uses the same full `ep_name` used
in reports and assignment, such as `CUDAExecutionProvider` or `DmlExecutionProvider`.

Dedicated adapters are internal compatibility details: CUDA uses `AppendExecutionProvider_CUDA_V2`; DirectML uses
the public `OrtSessionOptionsAppendExecutionProvider_DML` export required for the 1.24 path, disables memory patterns,
and accepts only a nonnegative integer string `device_id` (default `"0"`). Other supported built-ins use generic
`AppendExecutionProvider(ep_name, options)` with the canonical full name, not an alias from a second profile field.
Adapter availability and minimum-runtime behavior must be verified against ORT 1.24, not inferred from current-source
headers. Generic append is not universal; unsupported runtime/provider combinations fail explicitly. Actual
ORT 1.24.4 checks cover CPU execution and canonical `AzureExecutionProvider` generic append reaching the expected
CPU-fallback session rejection. DirectML's entry point resolves but device setup fails on this host; CUDA execution
is unchecked. These are separate from successful non-CPU inference, which remains outstanding.

Plugins register the library, enumerate `GetEpDevices()`, filter by `ep_name`, select the index, and append via
`AppendExecutionProvider_V2`. No matching device is an error, never CPU substitution or mass skip. Broader capability
declarations, tags, and tolerance overrides require future explicit contracts; they are not accepted pilot fields.

## Distribution

The required distribution is source-first: versioned standalone runner source plus an immutable generated case
archive, contracts, and provenance. EP implementers may compile the runner; they do not need an ORT source build,
private libraries, or producer tooling to consume the cases. Prebuilt runners may be offered for tested platforms,
but are not a universal release obligation. A possible kit layout is:

```text
onnxruntime-ep-op-conformance-<ORT_VERSION>/
  runner/
    CMakeLists.txt
    sources/
    include/
  cases/
    onnx/
    contrib/
  schemas/
  examples/
  LICENSE
  ThirdPartyNotices.txt
  VERSION
```

The source kit must include its public ORT header closure, standalone build files, dependency declarations, applicable
notices, and usage/contract documentation. Header discovery must be kit-relative, never an accidental fallback to an
ORT checkout. Building requires C++20, CMake, and public dependencies; Python may be needed for dependency generation
at build time. A built runner requires only cases, a profile, a compatible native runtime/provider, and their native
dependencies at execution. Offline dependency acquisition is not an initial requirement.

Runner binaries are platform-specific; generated cases and contracts should be platform-neutral. Source availability
does not establish portability on untested platforms. The initial source-kit CI target is Windows x64. Source and
case archives must be independently identifiable and record source/schema/corpus revisions, producer/build dependency
versions, hashes, and compatible runtime/API/operator requirements. API 24 is the runner minimum, not a promise
that a particular plugin built against newer headers works on ORT 1.24.

Packaging in PR CI should exercise the eventual release-preparation path, but ephemeral CI artifacts are not published
release kits. Release/nightly ownership, version policy, publication, and platform coverage remain separate follow-up
work. `tools/package_cases.py` implements reusable creation and verification of versioned archives from existing
JSON-generated or C++-exported case directories, preserving case/model/tensor bytes and recording hashes,
provenance, and model IR/opset requirements. It validates supported structure and dependency closure, not mathematical
correctness or EP compatibility. CI reuses its archive/hash mechanics for a separate, explicitly CI-only negative
fixture archive; public canonical-case commands do not bypass validation. `tools/create_cases_from_json.py` supports
fresh direct output or one managed build output. Unchanged inputs avoid regeneration; changed inputs replace the
whole generated set after successful generation, without retaining historical copies. Provenance and file hashes
remain recorded independently of directory naming. Source-kit/runtime packaging and bundle checks remain CI-only in `tools/ci/`;
a release installation/publication interface is not implemented.

An external EP should test against at least:

- The minimum ORT release it supports.
- The current ORT release used for packaging.
- An ORT `main` or nightly conformance kit as an early-warning lane.

## Dynamic plugin usage

A plugin profile supplies the library, registration name, canonical EP name, device index, options, and skips.
The pilot CLI is:

```powershell
onnxruntime_ep_op_conformance_test `
  --runtime .\runtime\onnxruntime.dll `
  --profile .\provider-profile.json `
  --cases .\cases `
  --report .\results.json
```

The registration name is chosen by the caller; the EP name must match the factory's reported identity. Plugin library
paths are resolved relative to the profile when not absolute. This illustrates the implemented interface, not a claim
of successful WebGPU plugin execution. The README includes a plugin-profile template.

## Deferred hosts

Browser/Wasm execution and a native static-linkage host/SDK are deferred. Future hosts must consume the same case
meaning and validator contracts without separately authored goldens or weaker result rules. The native pilot does
not validate ORT Web assembly, browser loading, static registration, or static/dynamic parity. Do not remove existing
browser coverage on the strength of a native-only pilot; host-specific continuity remains separately owned.

## Runner inputs and reports

The CLI takes exactly four option/value pairs: `--runtime` (ORT shared library), `--profile` (JSON target profile),
`--cases` (one case directory or a directory with immediate case children), and `--report` (JSON output path whose
parent exists). Discovery is not recursive; selection must be nonempty with unique case IDs. Runtime/provider
dependencies must be staged or discoverable. There is no dependency on authoring JSON or producer tools at execution.

| Report v1 field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `status` | Run `PASS`, `FAIL`, or setup/input/report `ERROR` |
| `runtime_version`, `profile` | Loaded ORT version and parsed target profile, when setup reaches those steps |
| `results` | One terminal result per selected valid case when setup/input validation permits execution |
| Result `id`, `status`, `duration_ms` | Stable case ID, `PASS`/`FAIL`/`SKIP`, elapsed milliseconds |
| Result `source` | Manifest provenance object retained unchanged, including on skips/failures |
| Result `reason`, `phase` | Skip/failure reason; failures identify `session`, `assignment`, or `execution_or_comparison` |
| Result `assignment` | Successful assignment evidence entries with `node`, `ep_name`, `operator`; not an internal hardware trace |
| `passed`, `failed`, `skipped` | Totals for a completed case loop; may be absent on run-level errors |
| `error` | Run-level diagnostic when an error prevents normal completion |

The runner prints the report and writes it when possible; report-write failure is an error, not a pass. Exit `0`
requires at least one pass and no failures; exit `1` means case failure or zero passes (including all skipped);
exit `2` means setup/input/report error. A profile skip does not avoid artifact validation. No device, corrupt
artifacts, unknown skips, empty selection, and invalid profiles cannot turn into a successful skipped run.

Producer provenance is currently `source: {definition}` for JSON generation, with the top-level `id` identifying the
authored entry, and
`source: {suite, test, file, line}` for C++ export. Keep source metadata with the artifact. These are not complete
reproducibility records or runtime source-file dependencies. Full reports/kits should additionally record suite and
schema revisions, source revision/local patch identity, model/tensor/profile hashes, generator version/seeds, runtime
and EP/plugin version/hash, selected device/environment, per-dataset/check evidence, and coverage selection. Unknown
EP provenance stays unknown, not fabricated. The CPU CI bundle's `prepared.json` records source/public-header/tool,
runtime, and staged-file hashes plus producer package versions; its consumer verifies staged-file hashes. Complete
generic-report hashes, device/plugin provenance, release metadata, richer summaries, and migration equivalence checks
remain future work; a selected-case PASS does not prove migration completeness.

## Dependencies and CI

| Stage | Dependencies and outputs |
| --- | --- |
| Author/build | JSON-to-artifact generation needs Python, NumPy, and ONNX; C++ capture needs the ORT-private test build. The proposed artifact-to-JSON converter is migration tooling. Native build needs C++20, CMake, public ORT headers, nlohmann JSON, and matching ONNX/protobuf headers and libraries. |
| Preparation | Generate immutable positive cases and separate derived negative fixtures. Package source, cases, profiles, provenance, and compatible native runtime/provider fixtures. Python and checkout access are permitted here. |
| Consumer build | Build the standalone packaged source outside an ORT checkout with public dependencies and kit-local headers. Compiler, CMake, Python, and hash-pinned dependency downloads are permitted. |
| Execution | Run the native executable against generated artifacts, runtime, and optional plugin. No source checkout, Python, generator, or private ORT test libraries are required. Dynamic native dependencies still must be present. |

### CI responsibilities

Existing EP workflows own ordinary conformance execution against their source-built runtime/provider in applicable
runtime lanes. They use their normal checkout, Python/build tools, native dependencies, and device provisioning;
neither staged-directory execution nor PATH isolation is required. CPU is a first-class target. Built-in and plugin
profiles use the same case bytes, assignment rules, and comparator. Do not add a central all-EP matrix duplicating
those workflows. Compilation-only lanes do not establish execution coverage.

Shared infrastructure regression checks run once in the designated source-build/infrastructure path, not in every
EP lane. These cover generation, validation, comparison, source-kit buildability, plugin setup, assignment/fallback
rejection, skip/error semantics, and minimum-runtime compatibility. A limited sample plugin is important here: CPU's
broad operator support cannot alone demonstrate that unsupported nodes fail rather than fall back.
Private exporter and shared producer regression checks stay in source CI, not external consumption. Workflow triggers
must route corpus, producer, runner, profile, and relevant public API changes to the applicable execution lanes.

### Shared build and case artifacts

The standalone CMake runner is exposed through `onnxruntime_BUILD_EP_OP_CONFORMANCE_TESTS=ON` as an opt-in ORT subproject.
Both modes build the same
sources and public API boundary. The parent build supplies public ONNX/protobuf/JSON targets and includes, without
private ORT libraries, a second dependency-fetch stack, or a second runner implementation. C++20 stays target-local.
Case generation and provider execution remain separate from executable definition.

The current runner requires dynamic native-library loading and ONNX model loading. Static-only and minimal runtimes
do not meet that contract; Emscripten requires a separate WebAssembly host. Cross-compilation and Android/iOS deployment
are not yet enabled or validated, rather than inherently incompatible with portable operator conformance.
Configuration errors distinguish these integration gaps from runner contract requirements; all remain rejected in
the current main-build path. Building the executable does not
register a provider run with CTest. Provider CI jobs invoke the runner directly with an explicit source-built runtime,
profile, generated cases, and writable report path. Comparator/exporter unit tests retain CTest integration.
Per-EP corpus execution does not invoke the full Python developer/negative harness.

Generate cases deterministically into the build tree once per workflow/input set, sharing exact bytes and metadata
between build/test jobs. Generation reuses the ORT build's Python environment and dependency policy; it does not use
an installed ORT wheel as an output oracle.
Track canonical definitions and producer inputs so changed inputs invalidate generation,
unchanged builds avoid regeneration, and stale or mixed cases cannot survive. Handle the generator's existing-output
refusal through fresh input-keyed directories or narrowly owned staging, never hand-edited generated files.
Negative fixtures remain separate from canonical positives. The producer uses the build's `Python_EXECUTABLE` and
records installed NumPy, ONNX, and protobuf versions without separately pinning or enforcing them. Inputs and producer
versions form a digest-named immutable directory and metadata under the build tree. Changed input or producer versions
select a fresh directory; an unchanged direct-target build does not regenerate. The dedicated CPU job invokes only
the native runner against the canonical corpus.

Independent EP workflows may initially regenerate using the same source revision and compatible producer dependencies,
recording their actual versions. PR changes use their own cases/generator, not a mutable latest-main archive or a cache
treated as authority. No
cross-workflow artifact service is required. Profiles contain setup and reasoned support exclusions, never alternate
goldens, models, tolerances, or validators. Retain reports/logs even on failure, with exact case selection, revisions,
hashes, runtime/profile identity, and available plugin/device provenance.

### Source-kit and plugin validation

Source-kit validation uses a matching shared runtime and limited sample plugin built from the same revision and
configuration. Reuse an existing source-build producer rather than adding a full ORT build solely for this fixture.
The external consumer depends on that producer and downloads narrow source-kit, case/scenario, and native-fixture
artifacts by immutable IDs. This publication and reusable consumer are implemented in the Windows CPU workflow DAG;
remote workflow execution remains pending.

The consumer has no ORT checkout and compiles the packaged source, not a staged prebuilt runner. Build-time tooling is
allowed; isolation is a focused final native-execution check with source/build directories unavailable and Python
removed from the execution environment. Verify artifact hashes/revision agreement and native dependency closure;
retain configure/build/test logs even on failure. Reuse existing canonical cases, fixture helpers, native comparison,
and scenario checks rather than maintaining a second corpus or validation framework.

Use an explicit bounded selection: the sample plugin supports static equal-shape float32 Mul, including scalar and
initializer cases, but not Add. Require known-supported IDs to pass with target assignment. Wrong expected Mul must
fail comparison; unskipped Add must fail session creation because CPU fallback is disabled. A reasoned Add skip must
coexist with mandatory supported passes. Invalid libraries/devices, malformed artifacts/profiles, unknown skip IDs,
and all-skipped selections must produce their intended failure or error, not merely any nonzero exit. Missing runtime
dependencies cannot satisfy a negative test. Do not auto-skip failures or require this fixture to support the growing
entire corpus.

Keep a small separate CPU smoke against pinned ORT 1.24.4 to prove the minimum API path. The current-source sample
plugin requests the current header API and must run only with its matching runtime, not be forced onto 1.24 by
weakening compatibility declarations. This integration smoke proves harness behavior, not real WebGPU or GPU use.

### Current CI implementation

[Windows CPU source CI](../../../../.github/workflows/windows_x64_release_build_x64_release.yml) enables the main-build
subproject and publishes separate build handoffs for two independent reusable workflows. The
[generic provider workflow](../../../../.github/workflows/windows-ep-op-conformance.yml), called with
`ep-name: CPUExecutionProvider`, receives the main-build runner/runtime,
exact canonical cases, CPU profile, and generation metadata. It runs the native executable directly, without checkout,
Python setup, the example plugin, or infrastructure tests, and retains its report even on failure.
Other Windows EP callers reuse the same bundle layout with a target-specific profile and compatible native libraries.
Inputs supply `ep-name` for job/report naming and runner labels; callers provide required hardware/drivers/dependencies.
The profile, not the display label, selects the EP. This does not add provider adapters or non-Windows host support.

The [infrastructure workflow](../../../../.github/workflows/windows-ep-op-conformance-infrastructure.yml) checks out the same
revision, tests generation/packaging and C++ export/replay using relocated main-build binaries, and publishes exact
positives, derived scenarios/profiles, runner source, and matching runtime/plugin fixtures. Source/case/fixture
manifests independently record revisions, versions, API, input/producer identity, and file hashes. Its no-checkout
source-kit job verifies agreement, builds packaged sources against kit-local headers and hash-pinned public dependencies,
and validates the native comparator. It stages bundles for a fresh execution job without source/build directories or
Python setup. PATH is restricted to System32. Matching-runtime plugin scenarios enforce two mandatory
Mul passes, comparison failure, unskipped Add session failure with fallback prohibited, reasoned skips, and intended
library/device/profile/artifact/skip errors. A separate bundle runs nine CPU checks on ORT 1.24.4; the current plugin
never runs against that older runtime. Package/generation provenance is retained with native reports; configure,
build, and test logs are uploaded even on failure. No remote workflow result is claimed. The Windows host supplies
PowerShell and the MSVC redistributable. CPU conformance and infrastructure validation depend only on the common ORT
build, not on each other's test results. Branch-protection owners must review their separate check names.

This proves bounded infrastructure, not release publication or WebGPU parity. The
[workstream](test_ownership_and_conformance_workstream.md#work-packages) owns deliverables, workflow changes,
dependencies, and completion checks. Its work packages are distinct from the migration phases below.

## Migration approach

### Phase 1: Shared tester and EP adoption

Land the bounded tester and CPU CI, then require all supported built-in EPs in applicable CI lanes. Use one canonical
corpus and explicit, reviewable support exclusions, not target-specific cases or inflated tolerances. Exercise CPU,
plugin setup, and an available non-plugin non-CPU EP with the same artifacts and checks. Include output mutations,
missing-provider/device failures, and assignment rejection. A sample plugin is harness evidence, not WebGPU evidence.

### Phase 2: Incremental generic C++ migration

Use the workflow above to migrate generic repository operator tests incrementally. The population includes generic
ONNX/contrib tests wherever authored, not just WebGPU directories. Account for parameter/loop variants, additions,
changed models/data/tolerances/settings/guards, and assertions throughout migration. Extend the tester only for
concrete needs with focused tests. Preserve existing coverage until every applicable consumer meets retirement.

Verify each portable replacement against the original C++ test's execution and assertions in every applicable EP
lane. CPU-only, optimized-away, bypassed, skipped, partial, or failing execution cannot stand in for required target
execution. Actual EP assignment and execution establish applicability, not a passing gtest that may bypass the EP.
Stable case IDs and reviewed semantic changes must survive reorderings and additions.

### Phase 3: WebGPU parity before extraction

Migrate all WebGPU-applicable generic tests with equivalent execution and applicable assertions in all applicable
WebGPU lanes, including isolated plugin execution for native plugin lanes. Account for additions and changes up to
extraction. Verify equivalent replacements for every C++ test in the WebGPU-applicable target set. Missing cases,
filter gaps, weakened checks, or skips caused by missing tester functionality block parity. Legitimate support
exclusions remain explicit and reviewable; they cannot erase existing WebGPU coverage. Successful export or a
converter round trip alone is insufficient. EP-specific and browser/Wasm continuity remain separate gates.

### Phase 4: Continue migration and publish kits

Continue broader repository generic-test migration after the WebGPU milestone. Publish immutable source-kit/case
previews, then compatible release/nightly kits with optional prebuilt runners for external CI. Migration completes
when every generic C++ test in the target set has an equivalent portable replacement in its applicable EP lanes,
including additions and changes during migration. Clean external runs use only the native runner and dependencies,
corpus/profile, runtime, and plugin, without private test binaries or Python. Kit publication alone proves neither
WebGPU parity nor repository migration completion.

## Validation status

The revised implementation uses C++20, executes once per dataset, rejects authoring/manifest `repeat_count`, accepts
two profile kinds, and uses only canonical `ep_name`. Local Windows checks pass: 23 exporter tests, 36 integration
checks without skips using the matching runtime/sample plugin and fresh C++ capture, and 10 comparator checks.
CPU and the sample plugin replay the same captured/data-defined artifacts; staged native runs work without Python
or source tools on PATH. Minimum-runtime CPU checks pass on ORT 1.24.4, with optional plugin/C++ tests reported as
skipped when not supplied. The pinned-dependency CPU CI bundle passes all nine local native scenarios; the workflow
has not been run on GitHub. Canonical Azure generic append on 1.24.4 reaches expected CPU-fallback rejection, not
successful non-CPU inference.

DirectML on ORT 1.24.4 failed setup because no usable display adapter was available. No successful existing non-plugin
non-CPU EP run or real WebGPU run is established. Assignment proves ORT target dispatch, not internal hardware use.
No existing generic C++ definition has been retired, converter implemented, full generic migration/WebGPU parity
established, or release kit published. Broader dtype/error/validator support and deferred hosts remain future work.

See the [pilot README](../../../../onnxruntime/test/ep_op_conformance/README.md) for commands to reproduce or extend
validation. Pilot completion is infrastructure evidence only, not extraction readiness.

## Remaining design work

- Identify the target generic C++ tests and applicable EP lanes, and verify migration equivalence against them.
- Add concrete-migration-required encodings/validators, negative-stage semantics, and explicit partial-assignment rules.
- Finalize schema evolution, provenance hashes, and deterministic conversion/generation checks.
- Define release artifacts, dependency packaging, large-data distribution, and later host integration separately.
