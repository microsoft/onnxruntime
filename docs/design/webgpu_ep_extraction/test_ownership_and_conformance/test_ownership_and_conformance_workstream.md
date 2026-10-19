# Workstream `test-conformance`: Test Ownership and Operator Conformance

Status: Working plan

[WebGPU EP extraction overview](../webgpu_ep_extraction.md)

## Objective

Preserve WebGPU regression coverage during extraction, assign every test to its long-term owner, and establish a
portable operator conformance layer that an external provider repository can run through public ORT interfaces.

The repository move must not wait for every reusable ORT test to be converted to a new format. It must wait until all
existing coverage is accounted for and continues to run against the isolated or external provider.

The detailed general-purpose conformance design is in
[Execution Provider Operator Conformance Suite](ep_operator_conformance_design.md).

## Terminology

| Term | Meaning |
| --- | --- |
| Conformance suite | The portable operator cases together with their execution and result semantics |
| Conformance kit | The versioned artifact published with an ORT release, containing cases, schemas, and a runner |
| Case archive | The platform-neutral conformance cases and schemas within a conformance kit |
| Provider profile | An EP's declared support surface, options, skips, and comparison overrides, supplied when running the conformance suite |

## Desired end state

- Every current WebGPU-related test has a stable owner and destination.
- No test disappears merely because its implementation moves repositories.
- Existing tests run against the plugin path before the legacy provider path is removed.
- WebGPU-specific implementation tests live with the provider.
- Portable operator correctness cases remain ORT-owned and are distributed in a versioned conformance kit.
- Generic plugin contract and ORT integration tests remain in ORT.
- Static and shared provider forms run equivalent operator cases and result rules.
- CPU fallback cannot produce a false pass.

## Test classification

Create an inventory covering C++, JavaScript/TypeScript, Python, package, browser, and CI-only tests. Assign each test
one primary class:

| Class | Long-term owner | Examples |
| --- | --- | --- |
| Portable operator conformance | ORT | ONNX and contrib operator semantics, type and shape combinations |
| WebGPU-specific implementation | WebGPU repository | Shaders, Dawn behavior, device limits, caches, provider options, and EP-specific operator behavior |
| ORT/plugin integration | ORT | Registration, session integration, loading errors, generic lifecycle |
| Host integration | Owning host repository | ORT Web module assembly, generic Node loading, package-host behavior |
| Temporary legacy | Explicitly recorded | Existing private ORT test retained while equivalent portable or provider-owned coverage is being established |

Each inventory entry should record:

- Current test and CI lane.
- Behavior protected.
- Current provider path.
- Classification and future owner.
- Planned destination or conformance case ID.
- Replacement status.
- Required platforms or devices.
- Tracking issue for temporary legacy coverage.

While JSEP and the native WebGPU EP coexist, `onnxruntime-web` runs one browser test list against both. The same
suite executes against JSEP in the default and `./all` bundles and against the native WebGPU EP in the `./webgpu` and
`./jspi` bundles, selected when the bundle is built. Which implementation a browser test exercises is therefore a
property of the CI lane, not of the test, and does not change the test's class, owner, or destination. Record the
provider path per lane, and treat the JSEP lane as following JSEP removal rather than this extraction.

Classification is a prerequisite for deleting or moving tests, not for beginning other workstreams.

A temporary legacy test is not a permanent ownership category. Its inventory entry must identify the intended final
class, replacement test or conformance case, tracking issue, and removal criteria. It remains blocking until the
replacement runs in all required lanes, after which the legacy test is removed.

## Regression protection before extraction

Before the source move:

- Run existing WebGPU operator tests through the plugin adapter path.
- Disable or detect CPU fallback for cases intended to validate WebGPU.
- Preserve current platform and browser lanes or document an approved replacement. Some WebGPU web lanes are
  currently non-blocking or build-only, so preserving them does not by itself establish a gate.
- Establish baseline results for the consumers listed in
  [WebGPU EP Repository Extraction](../webgpu_ep_extraction.md).
- Make isolated-tree tests blocking before removing their in-tree originals.
- Verify package installation and execution for existing Python, NuGet, Node, and Web consumers as applicable.

An existing private ORT test may remain temporarily authoritative if it executes the isolated provider. Conversion to
the portable conformance format can continue after extraction.

## WebGPU-specific test migration

Tests move with the provider when they validate implementation choices rather than portable operator semantics.
Likely categories include:

- WGSL generation and shader compilation.
- Dawn backend selection and integration.
- Device features, limits, and adapter behavior.
- Buffer, pipeline, and query caching.
- Provider options and diagnostics.
- `GPUDevice` and `GPUBuffer` WebGPU-specific interop behavior.
- Device loss and WebGPU-specific lifetime behavior.
- Performance regressions and implementation-specific workarounds.
- Plugin package contents and installation.

The `provider-isolation` workstream owns the physical relocation and external CI. This workstream defines the
classification and verifies that replacement coverage is equivalent.

## Conformance MVP

The initial conformance milestone should be deliberately bounded. It needs to prove the contract required for safe
extraction, not complete migration of ORT's operator test suite.

[Execution Provider Operator Conformance Suite](ep_operator_conformance_design.md) owns the runner, schemas, and
result semantics. This extraction requires that suite to deliver:

- A representative set of cases chosen from the current WebGPU support surface, with enough diversity to exercise
  capability discovery, model loading, execution, output comparison, profile-declared skips, and fallback detection.
- Execution of that set against WebGPU in both shared and static forms, checked against the expected outputs the
  cases carry.
- A versioned artifact the external provider repository can run without an ORT source checkout.

## Coverage continuity rules

- A test may be deleted only after its replacement is blocking in the required CI lanes.
- A moved test must protect the same behavior and platforms unless a reduction is explicitly approved.
- Forking a shared ORT helper transfers the behavior it implements to the provider. Where that behavior was covered
  only incidentally by tests of another consumer, the inventory entry must record whether existing coverage follows
  the fork or new provider-side coverage is required.
- Keep the inventory current while extraction is in progress. New WebGPU tests must be classified when added, and CI
  should detect test files or registrations missing from the inventory where practical.

## Work packages

1. **Inventory and classification:** enumerate tests and produce the ownership map.
2. **Plugin-path baseline:** run existing cases through the adapter and close fallback blind spots.
3. **WebGPU-specific relocation:** move provider-owned tests into the isolated staging root.
4. **Conformance schemas and runner:** implement the public execution and reporting contract.
5. **Representative case conversion:** convert a bounded extraction-gate set.
6. **External CI integration:** run existing and conformance coverage against clean provider artifacts.
7. **Coverage reporting:** detect newly skipped cases.

Inventory, runner design, and provider-specific relocation can proceed concurrently. Static runner validation depends
on the `plugin-boundary` workstream's static registration facility.

## Interfaces with other workstreams

### Plugin boundary and Web/Wasm integration

- Requires dynamic and static provider registration entry points.
- Supplies fallback detection and parity gates.
- Keeps generic plugin contract tests in ORT.

### Provider isolation and repository migration

- Supplies the test ownership map and required destinations.
- Requires standalone provider artifacts and CI environments.
- Moves provider profiles and implementation tests with the provider.

### Node plugin migration

- Supplies Node package installation and execution coverage for the consumer dispositions.
- Reuses portable cases where practical but keeps Node host-loading behavior in the Node workstream.

## Extraction gates

Extraction may proceed when:

- Every existing WebGPU-related test is inventoried and classified.
- Existing blocking behavior continues to run against the isolated provider.
- WebGPU-specific tests have moved or have blocking equivalent coverage.
- ORT integration tests cover dynamic and static registration contracts.
- The conformance MVP runs a representative set against shared and static WebGPU.
- CPU fallback produces a failure for cases requiring WebGPU assignment.
- Existing supported consumers pass installation and execution tests.
- Remaining temporary legacy tests have owners and removal criteria.

Complete conversion of all suitable `OpTester` and `ModelTester` cases is not an extraction gate. Loss of current
tested behavior is an extraction blocker.

## Completion criteria

- The ownership inventory contains no unclassified tests and remains current through the extraction cutover.
- The external provider CI protects WebGPU-specific behavior and current operator coverage.
- ORT publishes and consumes a usable conformance kit.
- Static and dynamic WebGPU reports are comparable and expose coverage regressions.
- Temporary legacy coverage is either removed or tracked with explicit exit criteria.
- Continued conformance expansion no longer requires coordinated provider source changes.

## Open questions

- What exact current test set defines the extraction regression baseline?
- What additional reporting is needed for cases that intentionally permit partial graph assignment?
- Which browser tests can consume the same portable cases without changing semantics?
- Which test generators should remain in ORT, move, or be copied?
