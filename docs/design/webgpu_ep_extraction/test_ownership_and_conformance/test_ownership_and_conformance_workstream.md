# Workstream `test-conformance`: Test Ownership and Operator Conformance

Status: Revised design reflecting the agreed pilot review and CI organization. The bounded implementation passes
local checks; isolated CPU CI is implemented and locally rehearsed, but has not run on GitHub. Main-build integration,
source-kit validation, and broader EP adoption are not implemented yet.

[WebGPU EP extraction overview](../webgpu_ep_extraction.md)

## Objective

Establish a provider-neutral corpus and public-API-only native tester, integrate all supported built-in EPs in
applicable CI lanes, and incrementally migrate the repository's generic operator C++ tests. CPU CI is the first
integration, not the end state. Plugin EPs use the same corpus, execution, and comparison through plugin setup.

Before WebGPU extraction, migrate **all WebGPU-applicable generic operator tests**, preserving equivalent execution
and relevant assertions in applicable WebGPU lanes. Export alone, a converter round trip, a representative subset,
or continued private test-binary execution does not satisfy that milestone. Missing tester functionality cannot be
turned into a skip and counted as migration. Broader repository migration continues after the WebGPU milestone.

The primary external audience is EP implementers. They may compile a standalone source kit without an ORT checkout,
ORT source build, or private libraries; build-time tools including Python are allowed. Native execution needs neither
Python nor the source kit. Readable JSON becomes each migrated case's single canonical source; generated
ONNX/tensor/manifest binaries remain derived outputs. Prebuilt runners are optional, not required on every platform.

EP-specific unit-test migration is **out of scope** and belongs to provider code migration. Classification may identify
such tests and their owners; this workstream does not relocate or reimplement them.

The detailed EP op conformance design is in
[Execution Provider Operator Conformance Suite](ep_operator_conformance_design.md).
See the [pilot README](../../../../onnxruntime/test/ep_op_conformance/README.md) for actual build and execution commands.

## Terminology

| Term | Meaning |
| --- | --- |
| EP op conformance suite | The portable operator cases together with their execution and result semantics |
| EP op conformance kit | Planned versioned distribution of standalone runner source, generated cases, contracts, and provenance; optional prebuilt runners |
| Runner source kit | Buildable standalone source, public header closure, dependency declarations, notices, documentation, and version/hash metadata; no private ORT libraries |
| Case archive | Immutable platform-neutral generated cases and associated contract/provenance metadata, independently identifiable from runner source |
| Provider profile | Target setup/options and explicit, reviewable case exclusions; no alternate goldens or pilot comparison overrides |
| Migration target set | The generic C++ operator tests to migrate, including their variants and assertions and additions/changes during migration |
| WebGPU generic parity | Migration of all WebGPU-applicable generic tests with equivalent execution/assertions in applicable WebGPU lanes |
| Preserved coverage | Equivalent execution and assertions to the original C++ test for every applicable existing EP consumer and CI lane |

Kit publication and complete migration are planned, not implemented pilot facilities.

## Desired end state

- All WebGPU-applicable generic test variants and assertions have equivalent migrated coverage before extraction.
- All supported built-in EPs run the shared tester in applicable CI lanes using one corpus with reviewable support
  exclusions. Plugins consume the same cases and validators. Broader generic-test migration remains the end target.
- ORT owns one corpus, loader, execution protocol, and portable comparator/validator implementation for CPU, legacy
  non-plugin EPs, and plugin EPs. Provider setup and declared support differ, not authored model/data/assertions.
- CPU is a first-class conformance subject, not only a reference generator; CPU and other existing consumers retain
  their applicable coverage when shared variants change canonical source.
- Each case has one canonical authored source: existing C++ until migration meets its retirement gate, then readable
  JSON generating the same artifact format. There are no permanently dual-maintained definitions or validators.
- Native dynamic runtime/plugin execution comes first; browser/Wasm and native static hosts are deferred.
- Non-CPU targets prohibit CPU fallback and verify named required nodes on the named EP through ORT 1.24+ public APIs.
- Existing EP workflows own provider execution; shared infrastructure checks run once rather than duplicating a
  central all-EP matrix. Ordinary EP lanes retain their normal environments; only external smoke execution is isolated.
- A future source-first versioned kit supports independent external CI. Source availability is not proof of tested
  portability. No source-kit packaging, release publication, or complete migration is claimed now.

## Test classification

The target set is the generic C++ operator tests in the repository, including additions and changes during migration
and shared ONNX/contrib cases under CPU/contrib directories, not just WebGPU directories. The tests themselves define
migration scope. Establish applicability from actual target assignment/execution, not a passing gtest that may have
bypassed the EP or run on CPU. Make support exclusions explicit and reviewable.

Record enough source/dependency/build/filter/seed/runtime/device/lane context to reproduce applicability and execution
evidence when verifying a replacement against its original C++ test.

Record parameter/loop variants, dtype/shape, initializer roles, attributes, session/run settings, lifecycle behavior,
custom verifiers, negative behavior, and assertions after `Run()`/`GetFetches()`. Classify ownership boundaries as needed:

| Class | Long-term owner | Examples |
| --- | --- | --- |
| Portable operator conformance | ORT | ONNX and contrib operator semantics, type and shape combinations |
| WebGPU-specific implementation | WebGPU repository | Shaders, Dawn behavior, device limits, caches, provider options, and EP-specific operator behavior |
| ORT/plugin integration | ORT | Registration, session integration, loading errors, generic lifecycle |
| Host integration | Owning host repository | ORT Web module assembly, generic Node loading, package-host behavior |
| Temporary C++ canonical source | ORT | Shared authored variant exporting derived artifacts while portable parity or data-authorship transfer is established |

Only portable generic-op migration is delivered by this workstream. The other classes identify responsibilities
of adjacent workstreams, not additional unit-test migration work packages here.

When reviewing a target C++ test and its replacement, check:

- Its parameter/loop variants, protected assertions, and corresponding portable cases.
- Applicable existing EP consumers, CI lanes, devices, and actual assignment/execution.
- Equivalent behavior and checks, including any assertions outside export.
- Explicit ownership for genuinely provider/host-specific assertions.
- Blocking replacement results before retirement; CPU-only, optimized-away, bypassed, skipped, partial, or failing
  execution cannot stand in for required target execution.

While JSEP and the native WebGPU EP coexist, `onnxruntime-web` runs one browser test list against both. The same
suite executes against JSEP in the default and `./all` bundles and against the native WebGPU EP in the `./webgpu` and
`./jspi` bundles, selected when the bundle is built. Which implementation a browser test exercises is therefore a
property of the CI lane, not of the test, and does not change the test's class, owner, or destination. Record the
provider path per lane, and treat the JSEP lane as following JSEP removal rather than this extraction.

Classification is a prerequisite for deleting or moving tests, not for beginning other workstreams.

A temporary C++ producer is not a second maintained corpus. Retire an authored variant only when every applicable
existing EP consumer/lane has equivalent coverage, not merely WebGPU. Preserve unrelated CPU/legacy variants until
they migrate. A thin in-tree consumer may reuse artifacts and shared validation; it must not copy definitions.

## C++ migration workflow

This pilot revision documents the process; it does not implement the converter or migrate existing cases:

The target C++ tests define the migration scope. Migration moves a test from C++ to the canonical conformance form
and removes the original C++ case. It requires equivalent execution and assertions in blocking CI for every applicable
existing EP consumer and lane. A retained C++ copy, including a placeholder, means the test is not fully migrated.

1. Export concrete ONNX models, inputs, authored expected outputs, explicit comparison policies, assignment
  requirements, and source provenance from the generic C++ variant. Continue original checks during validation.
2. Convert those artifacts to readable JSON for review/check-in, then generate executable model/tensor/manifest
  artifacts from JSON for the same native runner. JSON becomes the single canonical definition at transfer.
3. Preserve graph semantics, dtypes, initializer roles, assignment, and assertions. Reject anything unrepresentable;
  add tester functionality only for concrete migration needs, each with focused tests. Do not silently cast or drop.
4. Validate artifact-to-JSON-to-artifact round trips and equivalent execution/checks in applicable lanes. Audit custom
  verifiers and assertions outside export, such as `GetFetches()` checks and guards, before claiming equivalence.
5. Transfer canonical ownership to JSON and remove the original C++ case only after verifying equivalent replacement
  coverage in every applicable existing EP CI lane. Until then, keep C++ canonical and preserve its execution.
  Generated binaries are always derived, never a second maintained test corpus.

The [design](ep_operator_conformance_design.md#c-migration-workflow-proposed) specifies the acceptance criteria.
The current float32 single-node authoring subset cannot represent all generic tests; unsupported conversion is a
migration blocker, not completed coverage or a reason to weaken existing assertions.

## Regression protection before extraction

Before the shared-op extraction gate can pass:

- Account for all WebGPU-applicable generic tests and relevant assertions, including additions and changes.
- Migrate them with equivalent execution/checks in applicable WebGPU lanes; native plugin lanes exercise the isolated
  provider through the public plugin path. Keep browser/Wasm continuity separate where the native runner cannot cover it.
- Require named target assignment and prohibit CPU fallback. Optimized-away or unsupported target behavior is not PASS.
- Compare original and portable checks and execution protocols; an unexportable required assertion blocks parity.
- Require independent execution and equivalent replacements for every C++ test in the WebGPU-applicable target set,
  not just selected-case PASS.
- Account for source changes throughout migration, reviewing additions/removals, payloads/checks/tolerances, settings,
  guards, and support changes up to extraction.

Original C++ execution may support differential validation and remain canonical until its retirement gate. Export
alone is not migration and does not satisfy WebGPU parity. Do not weaken or skip required behavior to close the gate.
Browser, package, and implementation-specific continuity remains with its owning workstream; native pilot results do
not authorize reducing those lanes. Compilation-only or non-blocking lanes are not runtime conformance evidence.

## Scope boundary

WGSL/shader generation, Dawn behavior, caches, device limits/loss, WebGPU-specific interop, provider internals,
performance, package installation, and physical test relocation belong to provider/host migration workstreams. Record
an explicit owner when an assertion is classified there; do not silently discard it or claim it was portably replaced.

## Pilot versus extraction readiness

The bounded pilot proves infrastructure, not continuity. The revised contract uses a native C++20 runner with public
ORT headers, ONNX protobuf, and nlohmann JSON, dynamically loading an ORT runtime supporting API 24. Python
generates/tests data only.
CPU uses an ordinary session; legacy EPs use applicable public append APIs; plugin EPs use registration,
`GetEpDevices`, and `AppendExecutionProvider_V2`. Legacy provider DLLs are not automatically plugin API libraries.
Requested unavailable providers/devices fail explicitly; the pilot is not a universal adapter implementation.

The revised bounded contract covers inline dense float32, standard-domain successful cases, exact dtype/shape,
absolute-plus-relative comparison with explicit NaN/infinity rules, once-per-dataset execution, required named target
assignment, and no CPU fallback for non-CPU targets. Neither authoring nor manifest has `repeat_count`; the exporter
rejects `num_run_calls_ != 1`. Profiles use only `builtin`/`plugin` and canonical `ep_name`, with CPU implicit in the
built-in path and compatibility adapters internal. The opt-in `ConfigEpOpConformanceExport(directory, id)` captures once
before the provider loop and continues original execution; unsupported custom verifiers/errors/settings are rejected.

A small CPU-authored Mul export fixture and independent scalar Mul, initializer Mul, and broadcast Add data
definitions exist. They use one artifact loader/comparator. No existing shared authored variant has been retired;
neither full generic migration/WebGPU parity nor an existing variant's canonical-source transfer is established.

[CPU CI](../../../../.github/workflows/ep-op-conformance-cpu.yml) implements a standalone public-only CMake build,
Python preparation including incorrect-output/malformed-artifact/invalid-profile negatives, and a separate job that
downloads an immutable native bundle without checkout or Python setup. Its restricted-PATH consumer checks hashes,
intended failure phases/diagnostics, skip rules, and reports. All nine scenarios passed local rehearsal with the
CI-pinned ORT 1.24.4 runtime. This is the first lane toward all supported built-in EP adoption, not a claim that the
workflow has remotely passed. Windows system libraries, PowerShell, and the MSVC redistributable remain host prerequisites.

Revised local Windows results: 23 exporter tests, 36 integration checks without skips on the matching runtime/sample
plugin and fresh C++ capture, and 10 comparator checks pass. CPU/sample-plugin replay and staged runs without Python
are verified. Minimum-runtime CPU checks pass on ORT 1.24.4; canonical generic Azure append reaches the expected
CPU-fallback session rejection, not successful Azure inference. DirectML 1.24.4 failed device setup on this host; no
successful existing non-plugin non-CPU or real WebGPU run is established. The sample plugin validates the harness,
and assignment proves ORT dispatch rather than internal hardware use. Converter/migration, broader values/validators,
and release kits remain future work; static and browser/Wasm hosts remain separate.

## Coverage continuity rules

- A test may be deleted only after its replacement is blocking in the required CI lanes.
- A moved test must protect the same behavior and platforms unless a reduction is explicitly approved.
- Forking a shared ORT helper transfers the behavior it implements to the provider. Where that behavior was covered
  only incidentally by tests of another consumer, the migration review must establish whether existing coverage follows
  the fork or new provider-side coverage is required.
- Account for additions and changes to all generic operator tests throughout migration, identifying WebGPU
  applicability before extraction. CI should expose unaccounted tests/variants where practical.
- Required source variants/checks must all be selected, executed, and pass in each required native lane. Missing IDs,
  filtered/shard gaps, new skips, weaker checks, and unsupported exports block the gate even if selected cases pass.
- Stable case IDs identify semantic variants; content digests/revisions are separate. Source reorderings must not
  silently rename cases. Generated artifacts are immutable, never edited independently of the canonical source.
- Every C++ test in the applicable target set must have an equivalent replacement. Broader generic migration
  continues after the WebGPU milestone with existing non-migrated coverage intact.

## Work packages

The [design](ep_operator_conformance_design.md#dependencies-and-ci) defines CI architecture and requirements; these
packages define deliverables and dependencies. Dependencies below determine ordering; package numbers do not imply
a strict sequence and are distinct from the design's generic-test migration phases.

### 1. Main-build integration and CPU CI

**Deliverables:** expose the standalone runner through the main [CMake entry](../../../../cmake/CMakeLists.txt) and
enable it in the existing
[Windows x64 CPU source workflow](../../../../.github/workflows/windows_x64_release_build_x64_release.yml). Add
build-tree case generation and labelled CTest execution against its shared runtime with a builtin CPU profile.
Main-build switches and test registration are proposed, not existing command-line interfaces.

**Dependencies:** reuse the existing standalone project and parent-provided public dependency targets under the
[shared build and artifact requirements](ep_operator_conformance_design.md#shared-build-and-case-artifacts).

**Completion:** direct-target and normal workflow builds pass; CPU executes all applicable canonical IDs with blocking
assignment/output checks and retained reports. Verify no-change and changed-input rebuild behavior, rejection of
unsupported configurations, and immutable case bytes with revision/hash metadata ready for package 2.

### 2. Source-kit and plugin validation CI

**Deliverables:** reuse `build_x64_release` in that CPU workflow to publish narrow source-kit, case/scenario, and
matching runtime/`example_plugin_ep` fixture artifacts. Refactor the isolated
[CPU workflow](../../../../.github/workflows/ep-op-conformance-cpu.yml) into a reusable consumer, preserving required-check
names deliberately. Use the existing pilot generation/scenario helpers and the
[source-kit contents](ep_operator_conformance_design.md#distribution); do not add another full ORT build.

**Dependencies:** packaging may proceed alongside package 1, but consumption needs its exact positive artifacts and a
verified matching fixture. Call the consumer in the same CPU workflow DAG with a producer `needs` dependency and
immutable artifact IDs; `needs` cannot span independent workflow runs. The existing job does not publish these
artifacts today.

**Completion:** on Windows x64, build the packaged source without an ORT checkout and pass the
[plugin validation scenarios](ep_operator_conformance_design.md#source-kit-and-plugin-validation), including mandatory
supported cases, intended negative outcomes, a separate ORT 1.24.4 CPU smoke, and final isolated native execution.
Verify hashes, revisions, dependency closure, and retained build/run diagnostics. This proves source-kit consumption
and harness behavior, not release publication or WebGPU parity.

### 3. Existing EP CI adoption

**Deliverables:** integrate all supported built-in EPs incrementally in their existing applicable runtime lanes.
Start with these workflow changes, preserving normal dependencies and device setup:

| Workflow | Planned integration |
| --- | --- |
| [Windows CUDA](../../../../.github/workflows/windows_cuda.yml) | Build runner/generate cases before cleanup, preserve artifacts, and execute in the existing GPU test job with relocated CTest paths. |
| [Windows DML](../../../../.github/workflows/windows_dml.yml) | Run in the existing GPU build/test job using its source-built runtime and normal DirectML dependencies. |
| [Windows WebGPU](../../../../.github/workflows/windows_webgpu.yml) | Preserve runner/cases across the build/test split and execute on existing GPU test nodes; choose applicable variants rather than duplicating the full matrix automatically. |
| [Reusable Linux build](../../../../.github/workflows/reusable_linux_build.yml) | Pass explicit opt-in/profile inputs and preserve artifacts/permissions/path relocation in existing runtime lanes as needed. |

**Dependencies:** use package 1's shared build/generation/report contract once stable; adoption need not wait for
package 2's external-consumer validation. For each provider, establish its owner, runtime lanes/devices, shared-runtime
availability, public setup adapter, profile/options, reviewed exact-ID exclusions, and remaining work. Add adapters
only for concrete lane needs; deferred hosts remain separately owned.

**Completion per lane:** blocking canonical-case execution meets the design's
[CI requirements](ep_operator_conformance_design.md#ci-responsibilities) and
[report contract](ep_operator_conformance_design.md#runner-inputs-and-reports). Verify artifact handoff, target
assignment, fallback prohibition, required selections, and workflow triggers. Retain exact result IDs and provenance.
The [Linux WebGPU workflow](../../../../.github/workflows/linux_webgpu.yml) is currently build-only, not execution
evidence. Representative integrations do not complete all-EP adoption or the WebGPU parity gate.

### 4. Incremental generic migration

**Dependencies:** before migrating each case, package 3 must establish blocking replacement execution with equivalent
assertions in every applicable existing EP CI lane, including CPU through package 1. Export, conversion tooling, and
trial validation may proceed earlier as preparation, with the original C++ case remaining canonical and running.

Move each generic C++ test to the canonical conformance form through concrete export, readable JSON, generation,
and round-trip checks, removing the original C++ case as part of the move. Extend the tester for concrete needs with
focused tests.

### 5. WebGPU parity gate

Complete migration of all WebGPU-applicable generic tests before extraction, preserving equivalent execution and
assertions in applicable lanes and accounting for repository additions/changes. Export-only or missing-feature skips
do not count.
This gate depends on WebGPU lane adoption in package 3 and WebGPU-applicable migrations in package 4, not completion
of unrelated EP adoption or generic migration. EP-specific/browser continuity remains separate.

### 6. Broader migration and distribution

Broader generic migration may proceed alongside WebGPU work and continues after its parity milestone. Use the
validated packaging from package 2 as the release-preparation path for immutable source-kit/case previews and
compatible release/nightly kits, with optional prebuilt runners. Kit publication does not depend on completed
migration; publication and release ownership remain to be established.

This revision documents planned work, not its completion.
Publication or successful pilot subsets alone cannot close the extraction gate. Static hosts/SDK are deferred.

## Interfaces with other workstreams

### Plugin boundary and Web/Wasm integration

- `plugin-boundary` supplies registration, device discovery, and lifetime contracts for dynamic and static plugin EPs.
- `test-conformance` supplies portable cases and shared execution, assignment, fallback, and comparison requirements.
- The workstreams must agree how conformance executes statically registered EPs and which host-specific checks remain
  separate.

### Provider isolation and repository migration

- `test-conformance` supplies test classification and coverage-continuity requirements.
- `provider-isolation` relocates provider-owned tests and supplies compatible provider artifacts, test targets, and
  CI environments.

## Extraction gates

This workstream's shared-op gate is satisfied only when:

- All WebGPU-applicable generic operator tests and relevant assertions are migrated with equivalent execution in
  applicable WebGPU lanes, including isolated/external plugin execution in native plugin lanes.
- Target assignment and fallback prohibition are enforced. Exported artifacts alone or missing-tester-feature skips
  do not count as migration; missing variants, weakened checks, and unrepresentable required assertions block parity.
- Migration equivalence is verified against the WebGPU-applicable target C++ tests, including additions and changes
  through extraction.
- The external consumer executes independently of ORT source/private test binaries and Python at runtime.
- Each case has one canonical source, and every retired authored C++ variant retains equivalent coverage for all
  applicable existing EP consumers and lanes. Existing C++ remains intact until its retirement gate is met; merely
  exporting it does not satisfy the required WebGPU migration milestone.

These are future acceptance criteria, not claims of implemented gates. Separate provider/host migration gates still
apply. A subset/bridge cannot satisfy generic WebGPU parity. Unrelated non-WebGPU generic migration need not finish
before extraction, but remains in the broader migration target. Loss of applicable existing coverage blocks extraction.

## Completion criteria

- All supported built-in EPs run the shared tester in applicable CI lanes, using the same corpus and reviewable
  support exclusions; plugins share the machinery. CPU alone is not complete adoption.
- All WebGPU-applicable generic tests meet the parity milestone before extraction, with relevant assertions and
  equivalent applicable-lane execution; broader generic-test migration continues afterwards.
- Broader migration completes when every repository generic C++ operator test has an equivalent portable replacement
  for its applicable EP consumers and CI lanes, including additions and changes during migration.
- ORT publishes and consumes usable immutable runner source and generated case archives, with compatible runtime
  requirements and provenance recorded; prebuilt runners remain optional for tested platforms.
- CPU, legacy public-append targets, and plugin targets share case bytes and validators; no target-specific golden
  corpus or independently maintained validation path exists.
- Canonical-source transfers preserve every applicable existing consumer; JSON is the single maintained definition
  of each migrated case and binaries remain derived. Non-migrated C++ retains existing coverage.
- Migration verification rejects missing variants/assertions or reduced EP/lane execution, even if selected cases pass.

## Open questions

- Which applicability evidence and CI checks best expose additions and changes to the generic test population?
- Which concrete migrations next require dtype support, bounded validators, expected-error stages, or partial assignment?
- What release/schema compatibility and provenance contracts should preview and release kits enforce?
- Which later host integrations can reuse the portable contract without changing its meaning?
