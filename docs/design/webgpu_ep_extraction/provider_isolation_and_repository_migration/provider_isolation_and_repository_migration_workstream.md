# Workstream `provider-isolation`: Provider Isolation and Repository Migration

Status: Working plan

[WebGPU EP extraction overview](../webgpu_ep_extraction.md)

## Objective

Make the WebGPU EP an independently buildable and releasable component, first under an isolated staging root in the
ORT repository and then in a dedicated repository.

The repository move should become a controlled copy of an already independent subtree rather than a large refactor
performed at the same time as the move.

## Desired end state

The WebGPU repository owns:

- `OrtEpFactory` and `OrtEp` provider implementation.
- WebGPU kernels, contrib kernels, runtime, and device infrastructure.
- WebGPU-owned support code copied or replaced from ORT.
- Dawn selection, patches, and build configuration.
- WGSL templates, generators, and generated-source policy.
- WebGPU-specific unit, integration, browser, package, and regression tests.
- Shared and static provider build targets.
- Existing Python and NuGet plugin packaging and release pipelines.
- Version metadata, compatibility policy, CI, and release artifacts.

ORT should consume versioned artifacts or source and should not remain an implementation or packaging repository for
the provider.

## Isolation strategy

Expand the existing `plugin-ep-webgpu/` directory into the in-tree staging root. It already owns Python and NuGet
plugin packaging, version metadata, and release documentation. Add provider source, build, dependency, and test
ownership until the directory mirrors the intended external repository:

```text
plugin-ep-webgpu/
  cmake/
  include/
  src/
    ep/
    kernels/
    runtime/
    support/
  third_party/
    dawn/
  tools/
    wgsl/
  tests/
    unit/
    operators/
    integration/
    conformance/
  python/
  csharp/
  VERSION_NUMBER
  MIN_ONNXRUNTIME_VERSION
```

All provider-owned build inputs should be reachable from that root without consulting an implicit list of files
elsewhere in the ORT tree.

During isolation:

- Preserve selected history in the new repository through a filtered-history import, such as `git filter-repo` or an
  equivalent subtree export. Moving files into `plugin-ep-webgpu/` first makes the ownership boundary and subsequent
  history extraction clearer. The import should exclude unrelated ORT history where practical.
- Avoid changing behavior merely to change ownership.
- Keep ORT integration shims outside the provider root.
- Make generated files and downloaded dependencies explicit build outputs or inputs.
- Support an ORT build override pointing at an adjacent WebGPU checkout.

## Private dependency removal

Create a machine-reviewable inventory of:

- Private ORT headers included by provider sources.
- Private ORT libraries in provider link interfaces.
- Source files compiled into WebGPU targets from outside the provider root.
- Build variables and generated files supplied by ORT.
- Test-only dependencies on ORT internals.
- Platform and package scripts that assume an in-tree provider.

Classify each dependency:

| Resolution | Use when |
| --- | --- |
| Existing public API | The plugin SDK already expresses the required runtime interaction |
| New public plugin API | The operation is a stable, generally useful runtime boundary |
| WebGPU-owned copy | The code is implementation support and can evolve independently |
| WebGPU-specific replacement | Existing ORT code is unsuitable as a cross-repository dependency |
| ORT integration shim | The behavior adapts an ORT host or build to the external provider but is not part of provider behavior |

Examples of ORT integration shims include registering a statically linked factory during ORT Web startup, translating
an ORT reduced-operator configuration into provider build input, selecting a pinned provider source archive, or
adapting JavaScript-owned browser objects to the public bridge. These remain in ORT because they describe how an ORT
host consumes the provider.

Copied code must preserve license and provenance. Once copied, it becomes WebGPU-owned code and is not synchronized
with the ORT implementation.

## Standalone build contract

The isolated subtree should build against a declared ORT plugin SDK or installed ORT package without an ORT source
checkout.

It should produce:

- A native shared plugin library.
- A static plugin-API library for Emscripten and other static hosts.
- Provider metadata sufficient to diagnose ORT compatibility.
- Test executables or packages that consume public ORT interfaces.
- Development artifacts from the same commit as release artifacts.

The build should support:

- Pinned and overridable ORT SDK locations.
- Pinned Dawn and other third-party dependencies.
- Reduced-operator input from an ORT Web build.
- Platform-specific symbol visibility and export rules.
- Reproducible source archives.
- Adjacent-checkout development from ORT.

## Repository foundation

The external repository skeleton can be created before isolation is complete to validate:

- Directory and CMake layout.
- Required checks and platform matrix.
- Dependency caching and Dawn build time.
- Version and compatibility metadata.
- Artifact naming and retention.
- Issue ownership and contribution policy.
- Release automation.

The final code import should wait until a clean copy of the staging root builds and tests without undeclared ORT
source-tree dependencies.

## Packaging migration

Python and NuGet plugin packaging already exist. This workstream migrates their sources and release pipelines rather
than redesigning them.

The packaging model is:

- Core ORT packages do not depend on or bundle WebGPU.
- Users install the WebGPU plugin package separately.
- Plugin packages declare compatible ORT versions and fail clearly on incompatibility.
- Deprecated packages with built-in WebGPU are retired instead of being converted into plugin-dependent packages.
- `onnxruntime-web` consumes an immutable source archive or commit for static linkage.

The Node workstream consumes the native shared plugin artifacts produced here. Node package naming, plugin
registration, and compatibility behavior are owned end-to-end by
[Node plugin migration](../node_plugin_migration/node_plugin_migration_workstream.md).

## Test relocation

The `test-conformance` workstream classifies existing tests. This workstream physically moves tests classified as
WebGPU-owned, including:

- Kernel and shader generation tests.
- Dawn integration and backend tests.
- Device-limit and feature tests.
- Caching and provider-option tests.
- Browser interop tests for WebGPU-specific behavior.
- Package installation and artifact tests.
- Provider implementation regressions.

Portable operator cases and generic plugin contract tests remain owned by ORT. Temporary legacy tests may remain in
ORT until equivalent external coverage is blocking.

## Parallel work packages

1. **Dependency inventory:** enumerate includes, libraries, generated inputs, and build assumptions.
2. **Staging-root design:** define layout, targets, SDK inputs, and integration shims.
3. **Support-code isolation:** copy or replace implementation helpers.
4. **Dependency ownership:** move Dawn, patches, and WGSL generation.
5. **Standalone build:** produce static and shared artifacts outside the ORT build graph.
6. **Repository and CI scaffold:** validate clean-checkout development and release jobs.
7. **Packaging migration:** move the existing Python and NuGet pipelines.
8. **Source transfer:** copy the proven staging root and switch ORT to pinned consumption.

Inventory, repository scaffolding, packaging analysis, and test classification can begin immediately. Source transfer
depends on the standalone build.

## Interfaces with other workstreams

### Plugin boundary and Web/Wasm integration

- Private-dependency findings may create plugin API work.
- This workstream consumes public plugin EP API headers and static registration contracts.
- Browser-specific ownership must be agreed before moving bridge code.

### Test ownership and operator conformance

- The `test-conformance` workstream decides test ownership and minimum regression gates.
- This workstream supplies external test targets and CI environments.
- The provider profile, exclusions, and WebGPU-specific tests move with the provider.

### Node plugin migration

- This workstream supplies versioned native shared plugin artifacts.
- The Node workstream owns npm layout, loading, package naming, and user migration.

## Completion criteria

- All WebGPU-owned build inputs live under the isolated staging root.
- Shared and static libraries build from a clean copy without private ORT libraries or headers.
- Dawn and WGSL generation are provider-owned and reproducible.
- WebGPU-specific tests run from the isolated tree and remain blocking.
- Python and NuGet plugin packages are produced and tested from the isolated tree.
- The external repository CI builds, tests, and packages a clean checkout.
- ORT can consume a pinned external source artifact and can override it with an adjacent checkout.
- Native ORT packages remain WebGPU-independent.

## Versioning, provenance, and remaining questions

The WebGPU EP version is independent from the ORT version. Compatibility metadata declares the minimum and tested ORT
versions instead of coupling release numbers.

Package signing, provenance, and release controls should meet the same requirements as comparable ORT core packages.

ORT should consume WebGPU source using the standard mechanism used for comparable third-party source dependencies.
The dependency inventory should compare existing ORT mechanisms before selecting the exact implementation.

Remaining questions:

- Which copied ORT helpers need independent namespaces or API cleanup before transfer?
- How should reduced-operator configuration be represented as an external provider input?
- Which platforms and architectures are required for the first independent release?
