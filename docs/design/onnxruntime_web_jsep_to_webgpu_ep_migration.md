# Design: Migrate onnxruntime-web from JSEP to the native WebGPU EP

**Scope:** the `onnxruntime-web` JavaScript/TypeScript package (WebGPU backend; WebNN initialization glue) **and**
the native JS execution provider behind it — `onnxruntime/core/providers/js/`, `onnxruntime/contrib_ops/js/` and
their build plumbing — which Phase 2 also removes (§10.2). Roughly 130 C++/CMake files that a JS-only reading of
this document would hide.

**Deprecation notice:** [docs/JSEP_Deprecation.md](../JSEP_Deprecation.md) is the contributor-facing statement of
the freeze policy (bug and security fixes only), published as the first step of Phase 0 (§8). This document is the
implementation plan behind it.

**Related work:** [Remove the WebGL (onnxjs) backend from onnxruntime-web](onnxruntime_web_remove_webgl_backend.md)
— independent, but shares the `onnxruntime-web/all` bundle and the deprecation-warning utility.

---

## 1. Summary

`onnxruntime-web` implements WebGPU compute two ways:

- **JSEP** — the WebGPU compute path implemented in TypeScript over an Asyncify-compiled WASM core. Powers the
  default (`.`) and `./all` bundles.
- **Native WebGPU EP** — the C++ WebGPU execution provider compiled to WASM. Powers the `./webgpu` and `./jspi`
  bundles.

This document proposes removing JSEP and standardizing on the native WebGPU EP. The default bundle keeps the
`webgpu` backend key and public API while swapping implementation at build time; equivalent operator/type coverage
and model compatibility have not been established. WebNN
is unaffected: it is already the native C++ WebNN EP in both builds (both link `--use_webnn`; `session-options.ts`
selects the `WEBNN` EP either way). Removing JSEP only drops the shared JS init glue (`jsepInit('webnn', …)` →
`webnnInit(…)`, same `WebNNBackend`), not the WebNN EP itself.

- **Phase 0:** announce the deprecation and freeze JSEP to bug and security fixes, then close the build
  configuration, parity and CI gaps that block the flip. Add a full/unreduced native Asyncify configuration
  alongside the existing reduced native builds.
- **Phase 1 (a future release):** flip the default (`.`) and `./all` bundles to the full native Asyncify
  configuration, add a temporary, deprecated `onnxruntime-web/jsep` escape-hatch export preserving today's
  `/all` configuration, and ship warnings + docs. Validate the changed entry points in a prerelease package
  before stable; keep `/webgpu` and `/jspi` reduced and unchanged.
- **Phase 2 (a subsequent release after Phase 1):** delete JSEP — both the TypeScript backend and the native JS
  EP — and remove the temporary `/jsep` export and build flags.

The Phase 1 default flip requires its release gate to be met (§9), not a target release; preparatory changes can
land before that gate is met. Phase 2 begins when a review of reported parity gaps and JSEP usage concludes that
the temporary `/jsep` export can be withdrawn (§10) — at minimum one release after the stable Phase 1 switch.

### 1.1 Implementation status

**Status last reviewed:** 2026-11-10.

Use **Planned**, **In progress**, **Blocked**, or **Done**. On the default branch, **Done** means the changes are
merged and the required validation is complete. Once the other changes in an implementing PR are approved and
validation has passed, update its milestone to **Done** in that same PR, clear the remaining work, and retain the
PR link. This describes the proposed post-merge state and avoids a separate status-only follow-up. If required
validation can only happen after merge, keep the milestone **In progress** until that validation is complete.
Update the review date when updating the table. Keep progress updates here; the sections below define requirements
and decisions.

| Milestone | Status | Remaining work |
|---|---|---|
| Phase 0: deprecation announcement and contribution freeze | Done | |
| Phase 0: blocking native-WebGPU baseline CI | Done | |
| Phase 0: native-WebGPU IO-binding and proxy CI | Done ([#33842](https://github.com/microsoft/onnxruntime/pull/33842)) | |
| Phase 0: build configuration parity (§7) | In progress | A/B/C binary-size measurements and selected CPU/WASM compatibility checks are complete, not equivalent-coverage validation. Add full native Asyncify artifacts for the default and `/all`, preserving the reduced `/webgpu` and `/jspi` artifacts; validate build, loader, packaging, and CI selection in a separate PR. |
| Phase 0: GPU IO-binding parity (§8.2) | Planned | Establish zero-copy handoff and lifetime guarantees beyond correctness-suite coverage. |
| Phase 0: WebNN initialization coverage (§8.3) | Planned | Add and validate blocking coverage in a WebNN-capable browser environment. |
| Phase 0: deprecated WebGPU settings (§8.4) | In progress | Validate and merge initialization cleanup, custom-device coverage, and JSEP-only settings guidance. |
| Phase 0: JSPI browser coverage (§8.6) | Planned | Exercise the native JSPI path in a compatible browser. |
| Phase 1: compatibility export, warnings, and migration preparation (§9) | Planned | Rename the build flag, add deprecated `/jsep` preserving today's `/all` configuration, retarget operator docs, open the tracking issue, and prepare migration guidance and the usage baseline. |
| Phase 1: default flip (§9) | Blocked | Land an isolated default change and validate the changed default and `/all`, legacy `/jsep`, and unchanged reduced entry points in a prerelease package; meet the release gate before stable. |
| Phase 2: native profiling (§10.1) | Planned | Expose native profiling through the JavaScript API. |
| Phase 2: WebNN relocation (§10.1) | Planned | Move shared WebNN code out of the JSEP directory and update importers. |
| Phase 2: Windows/ADO native-WebGPU WASM builds (§8.6) | Planned | Provide GPU-capable replacement artifacts before removing JSEP build legs. |
| Phase 2: JSEP removal (§10) | Blocked | Ship stable Phase 1, retain the escape hatch for at least one post-switch release cycle, complete prerequisites, and review parity reports and usage before removal. |

---

## 2. Motivation

- **Remove duplicate maintenance.** JSEP and the native WebGPU EP implement the same operators twice (TypeScript
  and C++). The native WebGPU EP is the strategic direction and receives new op work.
- **Consistent runtime semantics.** The browser build shares kernel behavior with the rest of ONNX Runtime
  instead of maintaining a parallel TS implementation.
- **Simpler build matrix.** Removing JSEP deletes the temporary `USE_WEBGPU_EP` / `DISABLE_JSEP` build plumbing.

---

## 3. Goals and non-goals

### Goals

- Make the native WebGPU EP the default WebGPU backend, transparently for existing consumers (same import,
  `webgpu` key, and public API).
- Give JSEP consumers a low-effort migration path plus a safety net for the duration of the deprecation window.
- Remove the JSEP TypeScript compute path, the native JS execution provider, and their build variants.

### Non-goals

- **WebGL removal** — tracked in [onnxruntime_web_remove_webgl_backend.md](onnxruntime_web_remove_webgl_backend.md).
- No change to the Node binding, React-Native package, or ORT training-web.
- No new WebGPU features — this is a consolidation.

---

## 4. Background: backend selection

Whether a bundle uses JSEP or the native WebGPU EP is a build-time choice (distinct from the runtime
`executionProviders` selection), gated by the `BUILD_DEFS` flags `DISABLE_JSEP` and `DISABLE_WEBGPU` (native WebGPU
EP). The pivot is the temporary `USE_WEBGPU_EP` flag in `js/web/script/build.ts` (default `false`),
which sets `DISABLE_JSEP = !!USE_WEBGPU_EP` and `DISABLE_WEBGPU = !USE_WEBGPU_EP`. Both `USE_WEBGPU_EP` and
`USE_JSPI` are annotated in the source as temporary.

### 4.1 Current bundle / export map (`js/web/package.json`)

| Export | WebGPU impl | Async bridge | WASM binary suffix |
|---|---|---|---|
| `.` (default) | JSEP | Asyncify | `.jsep.wasm` |
| `./all` | JSEP | Asyncify | `.jsep.wasm` |
| `./webgpu` | native WebGPU EP | Asyncify | `.asyncify.wasm` |
| `./jspi` | native WebGPU EP | JSPI | `.jspi.wasm` |
| `./wasm` | — (CPU only) | — | `.wasm` |

---

## 5. Approach

- **Transparent default swap.** Both JSEP and the native WebGPU EP register under the `webgpu` key, so flipping
  the default bundle to the native WebGPU EP requires no consumer source changes.
- **Escape hatch.** Phase 1 adds a temporary `onnxruntime-web/jsep` export preserving today's full/unreduced
  JSEP-backed `/all` configuration, including WebGL while it remains supported. Make it available before or
  alongside the prerelease switch. It is an explicit application choice, not an automatic fallback, and stays
  available for at least one post-switch release cycle (§10). It is deprecated and warns once, doubling as a
  parity-bug funnel.
- **Coverage options.** The default and `/all` use the new full native Asyncify configuration. `/webgpu` and
  `/jspi` retain their existing reduced configurations. Revisit the separate reduced Asyncify binary later based
  on feedback and usage evidence; it can be consolidated onto the full binary without removing `/webgpu`.
  This is separate from JSEP removal and has no predetermined deadline.
- **Default and `/all` bundles.** Both switch to full native Asyncify, preserving their existing JavaScript
  backend selections. WebGL removal is tracked independently.

---

## 6. int64 handling

The native WebGPU EP reads int64 support from `ep.webgpuexecutionprovider.enableInt64` (`webgpu_provider_factory.cc`),
default `false`. It is reachable on `/webgpu` today via the untyped `extra` map
(`extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }`).

With int64 **off** (the default), the native WebGPU EP matches JSEP exactly: int64 arithmetic runs on CPU/WASM with full
`int64_t` precision, while int64 indices (`Gather` / `GatherElements` / `MaxPool`) run on the GPU with i32
truncation. `enableInt64 = 1` is an opt-in that additionally runs int64 arithmetic on the GPU. Kernels using
i32-truncated arithmetic are lossy for values outside `[-2^31, 2^31 - 1]`. The int64 `Abs`, `Neg`, `Relu`, and `Sign` kernels
preserve all 64 bits, but still require this opt-in. `transformers.js` already runs the native WebGPU EP
with int64 off at scale.

**Decision:** keep int64 off by default. `enableInt64 = 1` is an opt-in tradeoff, not needed for JSEP parity.
Optional follow-up: a typed `enableInt64?: boolean` option.

Graph capture forces int64 on (`enable_int64_{enable_graph_capture || enable_int64}` in
`webgpu_execution_provider.cc`) to keep the captured region all-GPU. So `enableGraphCapture = true` silently moves
int64 arithmetic to the GPU and truncates genuine `> 2³¹` values — an exception to default parity, flagged in the
migration guide (§12).

---

## 7. Build configuration parity

The JSEP and native-WebGPU WASM artifacts are not built with the same operator and type surface today, and the
difference is invisible from the JavaScript layer.

`.github/workflows/linux-wasm-ci-build-and-test-workflow.yml` defines a `reduced_size_build_args` set —
`--disable_ml_ops --disable_generation_ops --disable_types string float4 float8 optional sparsetensor
--include_ops_by_config onnxruntime/wasm/reduced_types.config --enable_reduced_operator_type_support` — and
applies it to the `.asyncify` and `.jspi` (native WebGPU EP) legs but **not** to the `.jsep` leg.

`onnxruntime/wasm/reduced_types.config` keeps every operator (`!no_ops_specified_means_all_ops_are_required`) and
restricts only types: the globally-allowed set is `bool, int8_t, uint8_t, int32_t, uint32_t, int64_t, uint64_t,
float, MLFloat16`, so `double`, `int16_t` and `uint16_t` are dropped. The `--disable_*` flags additionally remove
the `ai.onnx.ml` operators, the generation operators, and the string / float4 / float8 / optional / sparse-tensor
types.

**This is not a WebGPU-only concern.** The default bundle's `wasm` (CPU) backend runs on the same binary, so
flipping the default from `.jsep.wasm` to `.asyncify.wasm` *as currently built* would also narrow the operator and
type surface for CPU inference — a much broader change than swapping the WebGPU implementation, and one that fails
at load time rather than falling back.

Compare three build configurations:

| Build | Artifact | Reduced-size args |
|---|---|---|
| A | `.jsep.wasm` | no — today's default bundle |
| B | `.asyncify.wasm` | yes — today's `/webgpu` bundle |
| C | `.asyncify.wasm` | no |

`C − B` is the download cost of removing the native build's reduced-size restrictions; `B − A` and `C − A` are
the user-visible size changes. `C − A` does not isolate the cost of removing those restrictions: JSEP ships no C++
WebGPU kernels, while the native build
compiles the full WebGPU kernel set plus Dawn.

Report WASM-only and WASM-plus-loader measurements separately. Before drawing conclusions about overall consumer
download impact, also compare equivalent production JavaScript-plus-WASM totals: JSEP's JavaScript compute code is
not represented in the WASM measurements. Use the same entry-point scope, format, minification and compression
settings, account for actual HTTP response boundaries, and do not double-count embedded loaders.

**Decision (refined 2026-11-10):** use the full/unreduced native Asyncify configuration (C) for the default and
`/all` to avoid the reduced build's additional operator/type exclusions and bring coverage closer to today's
JSEP-backed configuration. Retain the reduced native Asyncify configuration (B) used by `/webgpu`.
Full/unreduced describes the build configuration, not verified coverage equivalence with JSEP or support for every
operator/type combination. Compatibility gaps will be assessed through prerelease testing and user reports, with
`/jsep` available during the migration window (§9–10). The planned menu is:

| Export | WebGPU implementation | Async bridge | Operator/type configuration |
|---|---|---|---|
| `.` (default), `./all` | native WebGPU EP | Asyncify | full/unreduced (C) |
| `./webgpu` | native WebGPU EP | Asyncify | existing reduced (B), unchanged |
| `./jspi` | native WebGPU EP | JSPI | existing reduced, unchanged |
| `./jsep` (temporary, deprecated) | JSEP | Asyncify | full/unreduced (A), preserving today's `/all` configuration |

This adds one GPU-capable WASM configuration compared with today. The full and reduced native builds share the
same kernel implementation, but each shipped configuration needs build, package-loading, and coverage checks.
The CPU-only `/wasm` artifact is outside this change.

**Artifact selection.** Full and reduced native Asyncify builds must have distinct WASM and `.mjs` loader
filenames and explicit bundle/loader routing. Keep the existing reduced `.asyncify.*` artifacts for `/webgpu`;
select and document the new full-native filenames in the build/packaging PR. Wire package exports, embedded and
external loaders, workers, CI artifact staging, and package-consumption tests consistently. The B/C measurements
below used separate build directories with the same `.asyncify.*` names; those names cannot distinguish two
variants shipped together.

Implement and validate this build/packaging change separately from the default switch. This decision does not
flip defaults or change build flags by itself.

**Measurement evidence (2026-11-09).** At commit `6372345db913822eff49b00556a0c268b735cb48`, all three Release
SIMD/threaded builds passed with WebNN, API exception catching and vcpkg enabled, and RTTI disabled. The WSL2
toolchain used Emscripten 4.0.23, CMake 4.2.3, Python 3.14.4, Node.js 22.16.0 and npm 10.9.2.

The following byte counts include a comparable production-default-scope, non-embedded IIFE JavaScript bundle, the
original Release `.mjs` loader, and WASM, compressed as separate HTTP responses:

| Build | Raw bytes | gzip bytes | Brotli bytes |
|---|---:|---:|---:|
| A: JSEP, unreduced | 30,711,146 | 7,211,122 | 4,376,976 |
| B: native, reduced | 30,073,459 | 7,514,495 | 4,630,893 |
| C: native, unreduced | 35,874,406 | 8,635,091 | 5,234,381 |

Removing the reduced-size restrictions (C minus B) costs 603,488 Brotli bytes (589.3 KiB, +13.03%) or 1,120,596
gzip bytes (+14.91%).
The overall C-minus-A increase is 857,405 Brotli bytes (+19.59%) or 1,423,969 gzip bytes (+19.75%). Compression used
gzip level 9 without timestamps/filenames and Brotli 1.2.1, quality 11, generic mode, window 22; all 18 compression
round-trips passed. These are measurements of this pinned toolchain, not published-package size guarantees.

**Targeted CPU/WASM compatibility tests.** Chromium 156.0.8078.4 passed all 18 focused browser assertions with
the `wasm` backend explicitly selected,
graph optimizations disabled, and two-thread initialization verified. A and C supported float32/float64/int16/uint16
`Add`, `StringConcat`, and `ai.onnx.ml::LinearRegressor`. B supported float32 `Add`; the other five cases failed
at session creation with the expected missing-kernel/operator errors. These checks establish support only for
the six selected cases; they do not establish equivalent operator/type coverage or general model compatibility
between JSEP and the full native build. They do not validate GPU, WebNN, JSPI, or package consumption.

---

## 8. Phase 0 — Announce and close gaps (before the flip)

Phase 0 is the pre-work, in three parts:

- **Announce.** Publish [docs/JSEP_Deprecation.md](../JSEP_Deprecation.md) and the in-tree README pointers in
  `onnxruntime/core/providers/js/` and `js/web/lib/wasm/jsep/`, so JSEP contributions can be redirected
  immediately. The maintenance policy must precede implementation work so new JSEP features do not accumulate
  during the migration.
- **Resolve the build-configuration question** (§7), which gates the flip.
- **Meet the parity and coverage requirements below.** Browser coverage must run in a real GPU/WebNN-capable
  environment; operator tests alone do not establish initialization, IO-binding or compatibility guarantees.

1. **Proxy-worker (`wasm.proxy = true`).** Blocking Release browser coverage must exercise the native WebGPU EP
   with Asyncify and proxy mode over the EP-agnostic `proxy-wrapper.ts`, including device initialization and
   threading inside a Worker. Use CPU-resident IO: proxy mode does not support GPU-resident inputs or
   preallocated outputs.
2. **IO-binding.** Blocking native-WebGPU browser coverage must exercise both `--io-binding=gpu-tensor` and
   `--io-binding=gpu-location`, including uploads, preallocated or ORT-allocated GPU outputs, readback and disposal.
   The native path must preserve zero-copy handoff and dispose/lifetime semantics. Passing correctness suites
   alone does not establish true zero-copy handoff or all lifetime guarantees.
3. **WebNN.** WebNN is already the native C++ WebNN EP in both builds (`session-options.ts` selects `WEBNN`
   either way); the flip only swaps the JS init bridge (`jsepInit('webnn', …)` → `webnnInit(…)`, same
   `WebNNBackend`). Blocking browser coverage must establish equivalent native `webnnInit` behavior, including
   session creation and inference, in a `navigator.ml`-capable environment.
4. **Global `env.webgpu.*` settings.** Do not extend deprecated adapter-selection settings to the native EP:
   - **`powerPreference` / `adapter` / `forceFallbackAdapter`** — these deprecated globals remain supported by
     JSEP but are ignored by the native path. They must be documented as JSEP-only, including in migration
     guidance before the default flip. The global power preference never selected the native EP's actual
     inference device; adding native support would prolong use of the deprecated API.
   - The native EP retains its existing `WGPUPowerPreference_HighPerformance` default in both WASM and
     non-WASM builds. JSEP leaves its power preference unspecified when unset. This difference is an explicit
     migration compatibility exception, not a change to the native EP's default.
   - **`device`** — `env.webgpu.device` is output-only; setting it does not select a device. Applications that
     need specific adapter-selection options must create their own adapter/device and supply the device through
     the per-session option,
     `executionProviders: [{ name: 'webgpu', device }]`, which `session-options.ts` already registers via
     `webgpuRegisterDevice`.
   - Native initialization must retain availability checks but leave adapter/device creation to the EP.
     Supplying a custom device must not trigger an additional `navigator.gpu.requestAdapter()` call.
5. **Profiling.** `env.webgpu.profiling.ondata` is a JSEP-only mechanism: a per-dispatch JavaScript callback
   receiving `kernelId` / `kernelType` / `kernelName` / `programName`, timestamps and input/output tensor metadata
   (`WebGpuProfilingDataV1` in `js/common/lib/env.ts`), falling back to `console.log` when unset. The native
   WebGPU EP has no equivalent hook. It has a richer ORT-framework profiler (`webgpu_profiler.{h,cc}`, gated on
   the session's `enableProfiling`), but its JSON trace is unreachable from JavaScript:
   `session-handler-inference.ts` leaves `startProfiling()` as a TODO on **both** paths, and `endProfiling()` in
   `wasm-core-impl.ts` frees the filename returned by `_OrtEndProfiling` without surfacing it.

   **Decision:** `ondata` is dropped and documented as JSEP-only. Mapping `env.webgpu.profiling.mode` (and the
   deprecated `profilingMode`) onto the session's `enableProfiling`, implementing `startProfiling()`, and
   surfacing the JSON trace are a **Phase 2 prerequisite** (§10.1), not a flip gate — the `/jsep` escape hatch
   preserves today's behavior for the whole deprecation window.
6. **CI configuration requirements.**
   - Before removing JSEP build legs, Windows/ADO WASM builds must provide native-WebGPU-capable replacement
     artifacts through `tools/ci_build/github/azure-pipelines/templates/win-wasm-ci.yml`.
   - Browser coverage must exercise the native JSPI path using `--jspi` on a JSPI-capable browser; building the
     `/jspi` bundle alone is insufficient.
   - Use consistent, explicit suite selection across GitHub Actions and ADO.
   - Build both full and reduced native Asyncify configurations with unambiguous artifact selection. Exercise
     native GPU correctness for both, and retain JSPI and JSEP coverage. Add focused capability checks for the
     full build and the reduced build's intentional exclusions.
   - Each test invocation must select its intended WebGPU implementation. Because `--webgpu-ep` rebuilds
     `dist/ort.all[.min].js` in place and `karma.conf.js` loads that path, test ordering and rebuilds must not
     cause a test to exercise the wrong implementation. Package-consumption tests must use an independently
     packed production bundle, not a development bundle overwritten by a test run.

Per-session typed options cover JSEP's option surface.

---

## 9. Phase 1 — Flip + deprecate (a future release)

1. **Rename the build flag.** `USE_WEBGPU_EP` / `--webgpu-ep` inverts in meaning once the native EP is the
   default. Replace it with a `--jsep` opt-in (`--no-jsep` selecting native). Land the rename while JSEP is still
   the default so it carries no behavior change.
2. **Escape hatch.** Add the temporary, deprecated `onnxruntime-web/jsep` export preserving today's full
   JSEP-backed `/all` configuration, including WebGL while supported. Ship it before or alongside the prerelease
   default switch, with package-managed resolution selecting the matching JSEP loader and WASM.
3. **Warn once.** The `/jsep` build warns once (respecting `env.logLevel`) via a shared deprecation-warning
   utility, also used by the WebGL effort; the native default emits nothing.
4. **Retarget the operator-doc generator.** `generate-webgpu-operator-md.ts` parses the JSEP registrations
   (`js_execution_provider.cc`, `js_contrib_kernels.cc`), so post-flip `webgpu-operators.md` describes the wrong
   EP; point it at the native WebGPU EP registrations (those JSEP sources are removed in Phase 2).
5. **Flip the default.** Build `.` and `./all` with the full native Asyncify configuration from §7, preserving
   their respective JavaScript backend selections. The `webgpu` key and public API are unchanged. `/all` keeps
   WebGL until the WebGL effort removes it; `/webgpu` and `/jspi` stay reduced and unchanged. Land the required
   build/packaging preparation first and keep the default switch an isolated, independently revertible commit.
6. **Open the tracking issue.** The public early-warning and parity-report channel: what changed, how to pin
   `/jsep`, and where to report native-WebGPU-EP gaps. Link it from the deprecation notice and the release notes.
7. **Docs.** Deprecation banners + migration guidance in `js/web/README` (hand-authored) and a release-notes entry
   covering the new full-native WASM/loader filenames (§7), `/jsep` pinning, full versus reduced coverage, and the
   int64 `extra` opt-in.
   This is the first user-facing announcement; capture the JSEP usage baseline (§11) before it ships.
8. **Prerelease validation.** Publish the changed default and `/all` together with `/jsep` in a prerelease
   package before promoting the switch to stable. Validate the actual packed production package, including
   supported import conditions, embedded/external loader selection, WASM and worker loading, and explicit
   `wasmPaths`. Verify native GPU execution, CPU fallback, and targeted operator/type cases; verify `/jsep`
   preserves the legacy configuration and `/webgpu` and `/jspi` remain reduced and functional. Confirm `/all` and
   `/jsep` retain WebGL while supported. Invite compatibility reports and resolve blocking regressions before
   stable; existing stable releases remain unchanged during this validation.

**Release gate.** Before releasing the switched default as stable:

- Implement and validate the full/reduced build and artifact-selection policy in §7.
- Establish blocking browser coverage for proxy mode, both GPU IO-binding modes, and WebNN initialization, and
  validate the zero-copy and dispose/lifetime parity requirements (§8.1–3). Passing operator tests alone is
  insufficient.
- Document deprecated WebGPU adapter-selection globals as JSEP-only and provide custom-device migration
  guidance; validate that native initialization ignores these globals and does not request an extra adapter
  for a supplied device (§8.4).
- Exercise the native JSPI path and meet the suite-selection and bundle-isolation requirements (§8.6).
- Complete the compatibility export, warnings, build-flag rename, operator-doc retargeting, tracking issue, and
  migration guidance described above. Capture the JSEP usage baseline before the user-facing announcement ships.
- Complete the prerelease package validation above, including the working `/jsep` escape hatch, and resolve
  blocking compatibility regressions before stable.

Native profiling, WebNN code relocation, and Windows/ADO replacement WASM builds are removal prerequisites, not
default-flip gates (§10.1, §8.6). They must be complete before the affected JSEP code or build legs are removed.

---

## 10. Phase 2 — Removal (a subsequent release after Phase 1)

The `/jsep` hatch stays available for **at least one full release cycle after the stable default switch**, with
removal shipping in a subsequent stable release. Prerelease availability does not replace this post-switch window.
The window is not fixed in advance: it closes at an explicit checkpoint that reviews parity regressions reported
against the tracking issue, the warn-once funnel, and the JSEP usage signal (per-file CDN statistics for JSEP
versus both full/reduced native Asyncify and native JSPI artifacts, a breakdown npm download totals cannot give).
Nothing in the code or the published package hard-codes a
removal release; extending the window retains the JSEP implementation, CI legs, and packaged compatibility artifacts.

### 10.1 Prerequisites

Both are non-destructive and must land before any deletion, so that the deletion diffs stay reviewable:

- **Profiling.** Map `env.webgpu.profiling.mode` (and the deprecated `profilingMode`) onto the session's
  `enableProfiling`, implement `startProfiling()`, and stop discarding the `_OrtEndProfiling` trace filename in
  `endProfiling()`. Without this, removing JSEP removes the only working WebGPU profiling in the package (§8.5).
- **Relocate WebNN** out of `js/web/lib/wasm/jsep/` (`backend-webnn.ts`, `webnn/`) to a neutral path, repointing
  importers including `test/test-runner.ts` and `test/unittests/pool-output-shape.ts`.

### 10.2 Web package

1. Drop JSEP WASM artifacts; update `build.ts` / `package.json`; remove the `--jsep` build flag. Keep the reduced
   `/webgpu` and `/jspi` configurations; consolidating the reduced Asyncify binary is a separate future decision.
2. Delete `BUILD_DEFS.DISABLE_JSEP` and the code it gates; simplify `index.ts`.
3. Remove `pre-jsep.js` glue; confirm `post-webgpu.js` / `post-webnn.js` cover initialization.
4. Remove the `onnxruntime-web/jsep` export. A lingering `import 'onnxruntime-web/jsep'` then fails with the
   native bundler error — an acceptable build-time failure on this temporary surface. The default `.` import is
   unaffected.
5. Drop the JSEP WASM build legs and the `build_jsep` / `BuildJsep` pipeline parameters.

### 10.3 Native JS EP removal

Phase 2 also removes the C++ half — roughly 130 files, almost entirely deletion:

- `onnxruntime/core/providers/js/` (~90 files) and `onnxruntime/contrib_ops/js/` (~30 files).
- `cmake/onnxruntime_providers_js.cmake`, plus the `USE_JSEP` plumbing in `cmake/CMakeLists.txt`,
  `onnxruntime_providers.cmake`, `onnxruntime_providers_cpu.cmake` (the unguarded
  `onnxruntime_js_contrib_ops_cc_srcs` glob), `onnxruntime_unittests.cmake` and `onnxruntime_webassembly.cmake`;
  and the `--use_jsep` argument in `tools/ci_build/build_args.py` / `build.py`.
- `onnxruntime/wasm/pre-jsep.js` and `js/build_jsep.bat`.
- The `USE_JSEP` guards in `onnxruntime/test/optimizer/graph_transform_test.cc` and
  `group_query_attention_pre_norm_fusion_test.cc`.

Two items need care:

- **`kJsExecutionProvider` is public C API surface.** It is declared in
  `include/onnxruntime/core/graph/constants.h`, named in `onnxruntime_c_api.h`, and referenced from
  `provider_registration.cc`, `get_execution_providers.cc`, `provider_factory_creators.h`, `session_state.cc`,
  `conv_activation_fusion.cc`, `graph_transformer_utils.cc` and `external_data_loader.h`. Removing it is an API
  change and needs an explicit release-note entry; no tombstone is planned.
- **`post-webnn.js` is suppressed under JSEP** in `cmake/onnxruntime_webassembly.cmake`, so removing JSEP changes
  which glue a WebNN build links. This needs a WebNN smoke test, not just a successful compile.

**Ordering.** CI must stop passing `--use_jsep` (§10.2) *before* this code is deleted, or the pipelines break on
the deletion commit.

---

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Undiscovered native-WebGPU-EP parity gap vs. JSEP | Medium | `/jsep` escape hatch + warn-once funnel; differential tests |
| int64 behavior change vs. JSEP | Low | None by default (native-off matches JSEP); `enableInt64 = 1` is an opt-in tradeoff |
| Proxy / IO-binding / WebNN regressions in the native-WebGPU build | Medium | Require blocking browser coverage and targeted parity validation before the flip (§8) |
| Adapter selection changes after the default flip | Medium | Document deprecated adapter-selection globals as JSEP-only and the native high-performance default; direct applications needing a specific adapter to the per-session `device` option — release gate (§8.4, §9) |
| `env.webgpu.profiling.ondata` has no native equivalent | Low | Documented as JSEP-only; native profiling wiring is a Phase 2 prerequisite (§8.5, §10.1) |
| Windows/ADO produces no WebGPU-EP WASM build | Medium | Provide native-WebGPU WASM artifacts through `win-wasm-ci.yml` before removing JSEP build legs (§8.6) |
| `post-webnn.js` linkage changes for WebNN once JSEP is removed | Medium | WebNN smoke test on the Phase 2 build, not just a compile check (§10.3) |
| Full/reduced artifact mix-up or filename change breaks loading / `wasmPaths` | Medium | Distinct full-native WASM/loader names, explicit routing, prerelease package tests, and documented pinned-asset migration (§7, §9) |
| No telemetry on JSEP adoption | Medium | Per-file CDN statistics for JSEP vs full/reduced native Asyncify and native JSPI artifacts (baseline captured before the user-facing announcement), plus the warn-once funnel and the escape hatch |

---

## 12. Planned Phase 1 migration guidance

This guidance applies when Phase 1 ships, not to the current default bundle. In particular, the temporary
`onnxruntime-web/jsep` export is a planned Phase 1 addition and is not yet available.

- **Default import (`onnxruntime-web`) and `onnxruntime-web/all`:** the import paths and `webgpu` backend key
  remain unchanged; the implementation becomes the native WebGPU EP.
  Consumers that pin WASM artifacts must still update the path (see `wasmPaths` below).
- **Lower-overhead build (`onnxruntime-web/jspi`):** on JSPI-capable browsers, prefer `./jspi` — the same native
  WebGPU EP with lower per-call overhead than Asyncify. It retains its existing reduced operator/type
  configuration, not the default bundle's unreduced configuration.
- **Smaller Asyncify build (`onnxruntime-web/webgpu`):** retains today's reduced native configuration. Choose it
  when its operator/type coverage is sufficient and a smaller download is preferred; it is not the full default
  artifact.
- **Need JSEP for now:** import `onnxruntime-web/jsep` (temporary, deprecated) to select the previous `/all`
  configuration explicitly. There is no automatic retry through JSEP. File an issue if the native WebGPU EP does
  not work for your model — those reports are what set the removal timeline.
- **Deprecated adapter-selection globals:** `env.webgpu.powerPreference`, `env.webgpu.adapter`, and
  `env.webgpu.forceFallbackAdapter` are JSEP-only and ignored by the native EP. Create an adapter with the desired
  options, create a device from it, and pass that device through `executionProviders: [{ name: 'webgpu', device }]`.
  When ORT creates the device, the native EP defaults to high performance; JSEP leaves the preference unspecified
  when unset.
- **WebGPU profiling:** `env.webgpu.profiling.ondata` is JSEP-only and has no native equivalent. Code that
  registers an `ondata` callback keeps working on `/jsep` but silently receives nothing on the default bundle
  (§8.5).
- **Operator and type coverage:** the default and `/all` will use the unreduced native Asyncify artifact to avoid
  the reduced configuration's additional exclusions, including for `wasm` (CPU), and bring coverage closer to the
  JSEP-backed build (§7). Equivalent coverage has not been established; the investigation checked only selected
  CPU/WASM cases. Matching build flags do not guarantee model compatibility; report gaps through the tracking
  issue and use `/jsep` if needed during the migration window. `/webgpu` and `/jspi` remain reduced, and CPU fallback
  cannot restore kernels compiled out of a reduced binary.
- **int64-heavy models:** no change needed — the default matches JSEP. Exceptions that move int64 arithmetic to the
  GPU (lossy for genuine `> 2³¹` values): `extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }`, and
  `enableGraphCapture = true` (forces int64 on; see §6).
- **`wasmPaths` / pinned artifacts:** the default and `/all` switch from `.jsep.*` to new full-native Asyncify
  artifacts (§7), not the existing reduced `.asyncify.*` pair. Publish the exact filenames with the implementation
  and update object-form `env.wasm.wasmPaths`, direct artifact imports, preload/CSP rules, and copied assets.
  `/jsep` continues to use the JSEP artifacts; `/webgpu` and `/jspi` keep their existing artifact names.

> A pinned tracking issue (link TBD) is the authoritative, continuously-updated migration guidance. Warnings,
> docs, and CHANGELOG entries link to it.
