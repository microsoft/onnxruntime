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
- WebGPU-specific unit, integration, package, and regression tests, and browser tests of provider behavior.
- Shared and static provider build targets.
- Existing Python and NuGet plugin packaging and release pipelines.
- Version metadata, compatibility policy, CI, and release artifacts.

ORT should consume versioned artifacts or source and should not remain an implementation or packaging repository for
the provider.

The deprecated JSEP TypeScript compute path in `js/web/lib/wasm/jsep/` is out of scope. It is being removed rather
than moved.

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

- Move files into `plugin-ep-webgpu/` first, which clarifies the ownership boundary and simplifies the later history
  extraction described in History migration.
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

The subtree build should produce:

- A native shared plugin library.
- A static plugin-API library for Emscripten and other static hosts.
- Test executables or packages that consume public ORT interfaces.
- Development artifacts from the same commit as release artifacts.

The build should support:

- Pinned and overridable ORT SDK locations.
- Pinned Dawn and other third-party dependencies.
- Reduced-operator input from an ORT Web build.
- Platform-specific symbol visibility and export rules.
- Reproducible source archives.
- Adjacent-checkout development from ORT.

Browser-hosted provider tests are an exception to the no-source-checkout rule. Static linkage requires building ORT
Web with the provider, so the WebGPU repository builds ORT Web from a pinned ORT source revision using the
adjacent-checkout override. ORT separately validates its pinned WebGPU revision as part of ORT Web release gating.

## Packaging migration

Python and NuGet plugin packaging sources already live under `plugin-ep-webgpu/`. What remains outside the staging
root is the pipeline definitions that drive them, currently under `tools/ci_build/`. This workstream moves those
rather than redesigning the packaging.

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

The `test-conformance` workstream owns test classification and gating policy. This workstream physically relocates
the tests it classifies as WebGPU-owned and supplies the targets and environments they need:

- Moving test sources and data into the staging root and then the external repository.
- Building provider test targets against a plugin SDK rather than the ORT build graph.
- Providing CI environments, devices, and browser hosts for the relocated lanes.
- Retaining in-tree originals until their relocated equivalents run.

The classification table, coverage continuity rules, and extraction gates are in
[Test ownership and operator conformance](../test_ownership_and_conformance/test_ownership_and_conformance_workstream.md).

## Repository foundation

The external repository skeleton can be created before isolation is complete to validate:

- Directory and CMake layout.
- Required checks and platform matrix.
- Dependency caching and Dawn build time.
- Version and compatibility metadata.
- Artifact naming and retention.
- Issue ownership and contribution policy.
- Release automation.

## History migration

Create a history-migration manifest that lists every current and historical path whose changes should be retained.
The implementation has moved across multiple ORT directories, and path filtering does not automatically follow every
rename. Filtering only the final `plugin-ep-webgpu/` subtree would therefore omit earlier provider history.

Use `git filter-repo` or equivalent tooling to:

- Select the complete historical path set.
- Remap those paths into the new repository layout.
- Retain relevant authors, dates, commit messages, branches, tags, and merge relationships where practical.
- Exclude unrelated ORT source and history.

The filtered import rewrites commit IDs. Pull requests, reviews, issues, and other GitHub metadata are not Git objects
and do not transfer with repository history. Record the source ORT repository, extraction commit, filtering command or
script, and path manifest in the new repository so commits can be traced back to their original context.

Perform and review a trial history import before the final source move. Verify representative files with `git log`
and blame, and confirm that the resulting repository does not contain unrelated or sensitive content.

## Versioning and provenance

The WebGPU EP version is independent from the ORT version. Compatibility metadata declares the minimum and tested ORT
versions instead of coupling release numbers.

Package signing, provenance, and release controls should meet the same requirements as comparable ORT core packages.

ORT should consume WebGPU source using the standard mechanism used for comparable third-party source dependencies.
The dependency inventory should compare existing ORT mechanisms before selecting the exact implementation.

## Work packages

1. **Dependency inventory:** enumerate includes, libraries, generated inputs, and build assumptions.
2. **Staging-root design:** define layout, targets, SDK inputs, and integration shims.
3. **Support-code isolation:** copy or replace implementation helpers.
4. **Dependency ownership:** move Dawn, patches, and WGSL generation.
5. **Standalone build:** produce static and shared artifacts outside the ORT build graph.
6. **Minimum-version validation:** run the provider against its declared `MIN_ONNXRUNTIME_VERSION` runtime so the
   floor is verified rather than claimed.
7. **Repository and CI scaffold:** validate clean-checkout development and release jobs.
8. **Packaging migration:** move the plugin packaging pipeline definitions into the staging root.
9. **Source transfer:** copy the proven staging root, import filtered history, and switch ORT to pinned consumption.

Sequencing:

- Dependency inventory, staging-root design, dependency ownership, and repository and CI scaffold can start
  immediately and proceed in parallel.
- Support-code isolation depends on the dependency inventory.
- The standalone build depends on staging-root design, support-code isolation, and dependency ownership.
- Minimum-version validation and packaging migration depend on the standalone build for their final form, but both
  can be prototyped against current in-tree artifacts.
- Source transfer is the final package and depends on all of the others.

## Interfaces with other workstreams

### Plugin boundary and Web/Wasm integration

- Private-dependency findings may create plugin API work.
- This workstream consumes public plugin EP API headers and static registration contracts.
- Browser-specific ownership must be agreed before moving bridge code.

### Test ownership and operator conformance

- The `test-conformance` workstream owns test classification and gating policy.
- This workstream supplies external test targets, CI environments, and browser hosts.

### Node plugin migration

- This workstream supplies versioned native shared plugin artifacts.
- The Node workstream owns npm layout, loading, package naming, and user migration.

## Completion criteria

### Isolation milestone

The staging root is ready for transfer when a clean copy of it builds, tests, and packages everything listed in
Desired end state, with no provider build input resolving outside the root and no private ORT headers or libraries in
the link interface.

### End state

The provider lives in its own repository:

- The external repository CI builds, tests, and packages a clean checkout.
- ORT can consume a pinned external source artifact and can override it with an adjacent checkout.
- ORT retains provider integration shims only, not provider implementation, build, or packaging inputs.
- Native ORT packages remain WebGPU-independent.

## Open questions

- Which copied ORT helpers need independent namespaces or API cleanup before transfer?
- How should reduced-operator configuration be represented as an external provider input?
- Which platforms and architectures are required for the first independent release?
