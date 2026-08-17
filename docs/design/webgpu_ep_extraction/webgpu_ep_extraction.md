# WebGPU EP Repository Extraction

Status: Working plan

## Goal

Move the WebGPU Execution Provider (EP) into its own repository without making ongoing WebGPU EP development slower or
more fragile.

The new repository should own the WebGPU EP implementation, dependencies, tests, optional plugin packages, release
process, and development workflow. ONNX Runtime (ORT) should consume versioned WebGPU EP artifacts or source rather
than remain the implementation repository.

The WebGPU EP remains optional for native ORT consumers. Core ORT packages should support plugin loading but must not
bundle or depend on the WebGPU plugin package. `onnxruntime-web` is the exception because it must statically link the
provider into its WebAssembly module.

## Guiding principles

- Use the public plugin EP interface (`OrtApi`, `OrtEpApi`, `OrtEpFactory`, and `OrtEp`) as the only runtime boundary
  between ORT and the WebGPU EP.
- Use the same provider implementation for dynamically loaded and statically linked builds.
- Keep the WebGPU EP independently buildable, testable, versioned, and releasable.
- A normal kernel change should be developed, tested, reviewed, and released in the WebGPU repository alone.
- Do not create an undocumented cross-repository C++ interface to private ORT implementation code.
- Treat copied ORT helpers as WebGPU-owned forks. Preserve provenance and license, but do not keep the copies
  synchronized with ORT.
- Keep cross-repository integration reproducible through pinned versions, compatibility checks, and CI.
- Preserve existing tested behavior and supported consumers throughout the move.

## Workstreams

The effort is divided into four workstreams:

| Identifier | Workstream | Primary outcome |
| --- | --- | --- |
| `plugin-boundary` | [Plugin boundary and Web/Wasm integration](plugin_boundary_and_web_integration/plugin_boundary_and_web_integration_workstream.md) | Static and dynamic builds use the same public plugin EP boundary, including the ORT Web browser bridge |
| `provider-isolation` | [Provider isolation and repository migration](provider_isolation_and_repository_migration/provider_isolation_and_repository_migration_workstream.md) | WebGPU-owned code, dependencies, tests, and existing plugin packaging move under `plugin-ep-webgpu/` and then to an independent repository |
| `test-conformance` | [Test ownership and operator conformance](test_ownership_and_conformance/test_ownership_and_conformance_workstream.md) | Existing coverage is preserved, every test has an owner, and portable conformance coverage protects the external provider |
| `node-migration` | [Node plugin migration](node_plugin_migration/node_plugin_migration_workstream.md) | Existing bundled Node WebGPU support is replaced by an explicitly consumable plugin without regressing current users |

The detailed reusable conformance-suite design is in
[Execution Provider Operator Conformance Suite](test_ownership_and_conformance/ep_operator_conformance_design.md).

## Workstream dependencies

The workstreams can make progress concurrently, with these convergence points:

- Provider isolation identifies private dependencies. A dependency becomes public API work only when an existing
  public API cannot express a necessary, stable runtime interaction and the proposed addition meets the high bar for
  a permanent plugin EP API.
- Test classification determines which tests move with the provider and which remain in ORT.
- Static plugin registration enables ORT Web parity and static conformance execution.
- The Node workstream consumes generic plugin loading from ORT and native WebGPU artifacts from the external provider.
- The final repository copy depends on `plugin-ep-webgpu/` being independently buildable.

```text
Plugin boundary ───────────────> ORT Web static integration
       │
       ├───────────────────────> Node plugin loading
       │
Provider isolation ────────────> External provider artifacts
       │                              │
       └──────────────────────────────> Node WebGPU package

Test classification ──────────> Provider test relocation
Conformance MVP ──────────────> Static/shared parity gates
```

## Extraction launch matrix

The extraction launch matrix records existing consumers that require an explicit disposition before built-in code is
removed. Platform and architecture details remain to be inventoried in the individual workstreams.

| Consumer or package | Launch disposition | Extraction requirement |
| --- | --- | --- |
| `onnxruntime-web` | Consume a pinned external WebGPU source revision through static plugin registration | Required |
| Python WebGPU plugin package | Move the existing optional plugin package and release pipeline | Required |
| WebGPU NuGet plugin package | Move the existing optional plugin package and release pipeline | Required |
| Node WebGPU support | Provide a tested replacement for the WebGPU implementation currently bundled in `onnxruntime-node`; final package naming is open | Required before bundled support is removed |
| Deprecated native packages with built-in WebGPU | Retire according to an explicit compatibility plan; do not make core packages depend on or bundle the plugin | Required disposition |
| New native hosts without current WebGPU support | Add later unless explicitly included in the launch matrix | Not an extraction prerequisite |

The matrix should be updated as the current package and platform inventory is completed. Any removal from the launch
scope requires an explicit compatibility decision rather than silently dropping support.

## Integrated delivery stages

### Stage 1: Establish baselines

- Inventory private dependencies, build inputs, packages, consumers, platforms, and tests.
- Record every current test's owner and intended destination.
- Establish correctness, package installation, WebAssembly size, and startup baselines.
- Confirm the launch matrix and platform coverage.

### Stage 2: Execute workstreams in parallel

- Add generic static plugin registration and run WebGPU through the plugin path.
- Consolidate provider-owned sources and build inputs under `plugin-ep-webgpu/`.
- Preserve current tests while implementing the conformance MVP.
- Scaffold the external repository and CI against the standalone build contract.
- Design and implement the Node plugin-loading and package transition.

### Stage 3: Prove extraction readiness

The source move is ready when:

- Static and dynamic forms execute the same provider implementation through the public plugin EP API.
- `plugin-ep-webgpu/` builds from a clean copy without private ORT implementation headers or libraries.
- Every existing test is classified and current tested behavior remains blocking.
- WebGPU-specific tests run from the isolated provider tree.
- A representative conformance set detects output failures, unsupported regressions, and CPU fallback.
- Existing Python and NuGet plugin packages can be produced from the isolated tree.
- Node has an approved package transition and a tested replacement path.
- `onnxruntime-web` passes browser integration, reduced-build, binary-size, and performance checks through static plugin
  registration.

### Stage 4: Move and switch consumers

- Import the isolated provider code and selected history into the new repository.
- Publish versioned native plugin artifacts and immutable source artifacts for static consumers.
- Move Python and NuGet plugin releases to the WebGPU repository.
- Pin the WebGPU revision used by `onnxruntime-web`.
- Execute the approved Node package transition.
- Retire deprecated built-in packages and remove obsolete implementation and packaging code from ORT.

## Success criteria

- WebGPU has one runtime boundary with ORT: the public plugin EP API.
- Static and shared builds execute the same provider implementation and operator tests.
- `plugin-ep-webgpu/` can be copied and built outside the ORT source tree.
- The external repository owns WebGPU code, dependencies, tests, packages, and releases.
- ORT updates its pinned WebGPU revision through a routine dependency update.
- `onnxruntime-web` preserves supported functionality and accepted size and performance characteristics.
- Native ORT packages remain usable without installing WebGPU.
- Existing Node WebGPU users have a documented and tested migration path.
- Compatibility failures produce clear build-time, registration-time, or package-time diagnostics.

## Cross-workstream decisions still open

- What public API gaps remain after private dependencies are classified?
- What is the stable browser boundary for JavaScript `GPUDevice` and `GPUBuffer` objects?
- Which ORT-standard dependency mechanism should consume the external WebGPU source for WebAssembly builds?
- What platform and architecture matrix is required for the first independent release?
- What compatibility window should the WebGPU EP promise across ORT releases?
- Should the Node transition keep the `onnxruntime-node` name or introduce a core-only package such as
  `onnxruntime-node-core`?
- How should coordinated ORT API and WebGPU provider changes be tested before either repository merges?
