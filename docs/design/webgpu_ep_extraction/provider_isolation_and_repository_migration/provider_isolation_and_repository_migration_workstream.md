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
- Existing Python and NuGet plugin packaging, CI, and release pipelines.
- Version metadata, compatibility policy, CI, and release artifacts.
- User-facing provider documentation, which onnxruntime.ai references rather than duplicates.
- WebGPU issues and pull requests.

ORT should consume versioned artifacts or source and should not remain an implementation or packaging repository for
the provider.

The deprecated JSEP TypeScript WebGPU compute path under `js/web/lib/wasm/jsep/` is out of scope. It is being removed
rather than moved. The WebNN code in that directory is unaffected by this work.

## Isolation strategy

Expand the existing `plugin-ep-webgpu/` directory into the in-tree staging root. It already owns Python and NuGet
plugin packaging, version metadata, and release documentation. Add provider source, build, dependency, and test
ownership until the directory mirrors the intended external repository:

```text
plugin-ep-webgpu/
  cmake/
    patches/
      dawn/
  include/
  src/
    ep/
    kernels/
    runtime/
    support/
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

Dawn keeps its current form: a pinned, fetched dependency with patches applied at build time. The staging root holds
the pin and the patches, not a vendored copy of the source.

All provider-owned build inputs should be reachable from that root without consulting an implicit list of files
elsewhere in the ORT tree.

During isolation:

- Move files into `plugin-ep-webgpu/` first, which clarifies the ownership boundary and simplifies the later history
  extraction described in History migration.
- Avoid changing behavior merely to change ownership.
- Keep ORT integration shims outside the provider root.
- Make generated files and downloaded dependencies explicit build outputs or inputs. WGSL generation remains a
  build-time step; the Python requirement it places on consuming builds is acceptable because ORT already requires
  Python.
- Support an ORT build override pointing at an adjacent WebGPU checkout.

## Private dependency removal

The provider uses ORT-internal code deliberately. While the implementation and the runtime live in one repository,
reusing ORT's kernel-authoring types and operator helpers avoids duplicating code that is already maintained next to
it. Moving the provider removes the condition that made that trade worthwhile: code it cannot reach from another
repository has to be replaced or copied before the subtree builds on its own. Unwinding the coupling is a
precondition of the move.

The dividing line is the public ORT API. Everything else the provider depends on has to be addressed, whether it is
private implementation code or a utility ORT offers but does not ship. Those dependencies fall into three groups with
different dispositions:

- **Framework surface** — the kernel-authoring types the provider is written against. Replacing these is the
  kernel-authoring foundation work in the code isolation package, not a copy decision.
- **Operator helpers** — parameter parsing, shape math, and similar utilities shared with the CPU and CUDA EPs today
  only because everything lives in one repository. Copying these is the intended outcome, and the provider owns the
  correctness of its copies afterwards.
- **Plugin EP implementation utilities** — `include/onnxruntime/ep/api.h`, `common.h`, and
  `get_capability_utils.h`. ORT offers these for plugin EP implementations to use, and they depend only on the public
  C API, gsl, and the standard library. They are not shipped in the released package, so they are not public API and
  the provider copies them like anything else in this section. The adapter headers under
  `include/onnxruntime/ep/adapter/` are a separate tier that the extraction retires rather than copies.

Create a machine-reviewable inventory of:

- Private ORT headers included by provider sources.
- Private ORT libraries in provider link interfaces.
- Source files compiled into WebGPU targets from outside the provider root.
- Test-only dependencies on ORT internals.

Classify each dependency:

| Resolution | Use when |
| --- | --- |
| Existing public API | The plugin EP API already expresses the required runtime interaction |
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

The isolated subtree should build against the public ORT headers and an installed or pinned ORT package, without an
ORT source checkout.

Inventory the build variables and generated files ORT's build currently supplies to the provider, since each one is
either reproduced by the subtree build or becomes an input it must be given.

The subtree build should produce:

- A native shared plugin library.
- A static plugin-API library for Emscripten and other static hosts.
- Test executables or packages that consume public ORT interfaces.
- Development artifacts from the same commit as release artifacts.

The build should support:

- Pinned and overridable ORT package and header locations.
- Pinned Dawn and other third-party dependencies.
- Reduced-operator input from an ORT Web build.
- Platform-specific symbol visibility and export rules.
- Reproducible source archives.
- Adjacent-checkout development from ORT.

Browser-hosted provider tests are an exception to the no-source-checkout rule. Static linkage requires building ORT
Web with the provider, so the WebGPU repository builds ORT Web from a pinned ORT source revision using the
adjacent-checkout override. That revision is selected by the cross-repository version policy below. ORT separately
validates its pinned WebGPU revision as part of ORT Web release gating.

## Pipeline and packaging migration

Python and NuGet plugin packaging sources already live under `plugin-ep-webgpu/`. The plugin build, test, and
packaging pipeline definitions are still outside the staging root, split between
`tools/ci_build/github/azure-pipelines/` and `.github/workflows/`. This workstream relocates those and rewires them
to invoke the standalone build, but does not redesign the packaging scripts themselves.

The pipelines invoke `tools/ci_build/build.py --use_webgpu shared_lib` and consume artifacts from ORT's build output
locations, so they break the moment the standalone build replaces the in-tree one. The rewiring therefore lands with
the standalone build rather than after it, and no temporary shim over `build.py` is maintained. Relocating the
pipeline files is mechanical and happens earlier, with the rest of the staging-root move.

ORT-root packaging scripts reference WebGPU independently of the plugin packages, such as `setup.py` selecting the
retired `onnxruntime-webgpu` package name from a `--use_webgpu` flag. Inventory these alongside the pipelines. They
belong to the retired built-in package rather than to the plugin, so they are removed with it rather than relocated.

Most ORT CI lanes live under `.github/workflows/`, and the WebGPU ones do not share a single disposition. Lanes that
build the provider statically into ORT validate a configuration that stops existing, so they are retired rather than
rewired. `onnxruntime-web` is the exception: it keeps static linkage, against the pinned external source, along with
the WebAssembly build and browser-test lanes that serve it. Provider-owned concerns move to the WebGPU repository,
including external-Dawn validation, WGSL shader-key validation together with its action at
`.github/actions/webgpu-validate-shader-key`, and the plugin shared-library build.

The `plugin-ep-webgpu/rel-*` branch prefix exists only because WebGPU release branches share the ORT repository.
Pipelines that stay in ORT drop that trigger entirely, and pipelines that move use ordinary release branches in the
WebGPU repository.

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
- Building provider test targets against the plugin EP API and an installed ORT package rather than the ORT build
  graph.
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
- Component governance registration for Dawn and other dependencies, currently under `cgmanifests/webgpu/`.
- Security review, signing, and compliant release pipelines.

The compliance items are long-lead in practice and are easy to defer until they block a release. Establishing them
with the skeleton keeps them off the critical path.

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

After the cutover, WebGPU issues and pull requests belong in the new repository. Items open in `microsoft/onnxruntime`
at transfer time need a disposition: those describing provider behavior move or are refiled, while those describing
ORT-side integration stay where they are.

Perform and review a trial history import before the final source move. Verify representative files with `git log`
and blame, and confirm that the resulting repository does not contain unrelated or sensitive content.

## Versioning and provenance

The WebGPU EP version is independent from the ORT version. Compatibility metadata declares the minimum and tested ORT
versions instead of coupling release numbers.

Package signing, provenance, and release controls should meet the same requirements as comparable ORT core packages.

ORT should consume WebGPU source using the standard mechanism used for comparable third-party source dependencies.
The dependency inventory should compare existing ORT mechanisms before selecting the exact implementation.

## Cross-repository version policy

Each repository pins the other, so the pins must be arranged so that the dependency does not become circular. Only
one lane floats, and it never blocks a merge:

| Lane | Built or run against | Blocking |
| --- | --- | --- |
| WebGPU build | A released ORT version providing every EP API feature the provider references | Yes |
| WebGPU minimum-version validation | The declared `MIN_ONNXRUNTIME_VERSION` runtime | Yes |
| WebGPU browser tests | ORT Web built from the same released ORT revision as the build lane, consumed as source rather than as a package | Yes |
| WebGPU integration | ORT main | No |
| ORT | Its pinned WebGPU revision | Yes |

The blocking WebGPU lanes all target immutable released ORT artifacts — a package for the build and minimum-version
lanes, a source revision for browser tests — so none of them can wait on an ORT revision that does not yet exist. ORT
advances its WebGPU pin on its own schedule and declines the update when it fails.

Two ORT versions matter here, and they are not the same number:

- The **build-against version** must declare every EP API the provider references, including calls reached only
  through a runtime version gate, because a gated call still needs its declaration to compile.
- The **runtime floor**, `MIN_ONNXRUNTIME_VERSION`, is the oldest runtime the provider loads against. It can be
  lower, because newer calls are gated on the ORT API version detected at runtime.

They coincide only while nothing is gated above the floor. A build against newer headers cannot detect a mis-gated
call, so the floor has to be exercised by running against it. That is the minimum-version validation package.

The non-blocking integration lane against ORT main exists to catch ORT changes that break the provider while the fix
is still cheap. Neither blocking lane can do this: both target already-released ORT versions, so a regression
introduced on main stays invisible until it ships.

A failure in the integration lane is an ORT compatibility regression and is fixed in ORT, because the boundary is the
public plugin EP API. The exception is a provider dependency on unspecified behavior, which is fixed in the WebGPU
repository. Without a stated owner the lane goes permanently red and stops being read.

Adopting a newly added EP API therefore requires an ORT release carrying it before the provider can build against it.
This does not force the runtime floor upward, since the new call can be gated. The wait is a scheduling cost rather
than a deadlock: both repositories keep landing changes while it elapses, and the open question about consuming an
ORT pre-release exists to shorten it.

## Work packages

1. **Dependency inventory:** enumerate includes, libraries, generated inputs, and build assumptions.
2. **Staging-root design:** define layout, targets, ORT package inputs, and integration shims.
3. **Staging-root move:** relocate provider sources, Dawn patches, the WGSL templates and generator, and the plugin
   packaging and CI pipeline definitions into the staging root without changing behavior.
4. **Code isolation and standalone build:** replace the kernel-authoring foundation with WebGPU-owned equivalents,
   copy or replace the remaining implementation helpers, take over the Dawn dependency pin and fetch, produce static
   and shared artifacts outside the ORT build graph, and rewire the packaging and CI pipelines onto that build.
5. **Minimum-version validation:** run the provider against its declared `MIN_ONNXRUNTIME_VERSION` runtime so the
   floor is verified rather than claimed.
6. **Repository and CI scaffold:** validate clean-checkout development and release jobs, including component
   governance and compliance registration for the new repository.
7. **Source transfer:** copy the proven staging root, import filtered history, and switch ORT to pinned consumption.

The staging-root move is deliberately separate and mechanical. It is one behavior-preserving relocation that
conflicts with in-flight WebGPU changes exactly once, and it lets later contributions land in the destination rather
than adding to the isolation work.

Code isolation and the standalone build are one package because the provider is not independently buildable until the
kernel-authoring foundation is replaced, and that replacement is most of the work. The packaging and CI pipelines are
rewired in the same package because they drive the in-tree build directly and would otherwise break. Until the
separate build exists, isolation progress is visible in the WebGPU target's include and link lists in
`cmake/onnxruntime_providers_webgpu.cmake`; those lists are expected to shrink monotonically.

Sequencing:

- Dependency inventory, staging-root design, and repository and CI scaffold can start immediately and proceed in
  parallel.
- The staging-root move depends on staging-root design. It is not on the critical path, because relocating files does
  not change which base classes the provider uses and the existing static build keeps working from the new location.
- Code isolation and standalone build depends on the dependency inventory and the staging-root move. It also depends
  on generic static plugin registration from the `plugin-boundary` workstream, because the staging root must serve
  the static build before the adapter can be removed.
- Minimum-version validation depends on code isolation and standalone build for its final form, but can be
  prototyped against current in-tree artifacts.
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
Desired end state, with every build input either inside the root or a declared external dependency, and no private
ORT headers or libraries in the link interface. The ORT package and the pinned third-party dependencies are declared
external inputs, not exceptions to the milestone.

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
- Which ORT-standard dependency mechanism should consume the external WebGPU source for WebAssembly builds?
- What compatibility window should the WebGPU EP promise across ORT releases?
- May the WebGPU build-against ORT version reference a pre-release, to shorten the wait for a newly added EP API?
