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
`webgpu` backend key and swaps implementation at build time, so the change is transparent for existing code. WebNN
is unaffected: it is already the native C++ WebNN EP in both builds (both link `--use_webnn`; `session-options.ts`
selects the `WEBNN` EP either way). Removing JSEP only drops the shared JS init glue (`jsepInit('webnn', …)` →
`webnnInit(…)`, same `WebNNBackend`), not the WebNN EP itself.

- **Phase 0 (now):** announce the deprecation and freeze JSEP to bug and security fixes, then close the build
  configuration, parity and CI gaps that block the flip.
- **Phase 1 (a future release):** flip the default (`.`) and `./all` bundles to the native WebGPU EP, add a temporary
  `onnxruntime-web/jsep` escape-hatch export, and ship deprecation warnings + docs.
- **Phase 2 (a subsequent release after Phase 1):** delete JSEP — both the TypeScript backend and the native JS
  EP — and remove the temporary `/jsep` export and build flags.

Only Phase 0 is scheduled. Phase 1 begins when its release gate is met (§9), not at a target release. Phase 2
begins when a review of reported parity gaps and JSEP usage concludes that the temporary `/jsep` export can be
withdrawn (§10) — at minimum one release after Phase 1.

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
- **Escape hatch.** Phase 1 adds a temporary `onnxruntime-web/jsep` export, built with JSEP selected, that pins
  JSEP for the duration of the deprecation window — at least one release, closed at an explicit checkpoint rather
  than on a fixed date (§10). It is deprecated and warns once, doubling as a parity-bug funnel.
- **`/all` bundle.** `/all` bundles the WebGPU/WebNN backend and WebGL — two independent things changed by two
  efforts (this one flips WebGPU to native; the WebGL effort drops WebGL). It is kept to avoid breaking
  imports. Once both land, `/all` becomes a real alias of the webgpu/default artifact (`.asyncify.wasm`). The two
  efforts may land in either order; whichever lands second performs the repoint (§10, WebGL doc §8).

---

## 6. int64 handling

The native WebGPU EP reads int64 support from `ep.webgpuexecutionprovider.enableInt64` (`webgpu_provider_factory.cc`),
default `false`. It is reachable on `/webgpu` today via the untyped `extra` map
(`extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }`).

With int64 **off** (the default), the native WebGPU EP matches JSEP exactly: int64 arithmetic runs on CPU/WASM with full
`int64_t` precision, while int64 indices (`Gather` / `GatherElements` / `MaxPool`) run on the GPU with i32
truncation. `enableInt64 = 1` is an opt-in that additionally runs int64 arithmetic on the GPU — faster, but
i32-truncated and therefore lossy for genuine `> 2³¹` values. `transformers.js` already runs the native WebGPU EP
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

Three builds settle it:

| Build | Artifact | Reduced-size args |
|---|---|---|
| A | `.jsep.wasm` | no — today's default bundle |
| B | `.asyncify.wasm` | yes — today's `/webgpu` bundle |
| C | `.asyncify.wasm` | no |

`C − B` is the download cost of restoring full op/type parity; `B − A` and `C − A` are the user-visible size
change. `A` vs `C` is confounded and should not be read as the cost of the migration: JSEP ships no C++ WebGPU
kernels, while the native build compiles the full WebGPU kernel set plus Dawn.

**Decision:** either drop the reduced-size args from the default artifact (parity, larger download) or apply them
uniformly and document the narrowed surface as a migration-visible change. Resolve from the measurement before the
flip; record the outcome here and in the migration guide (§12).

---

## 8. Phase 0 — Announce and close gaps (before the flip)

Phase 0 is the pre-work, in three parts:

- **Announce.** Publish [docs/JSEP_Deprecation.md](../JSEP_Deprecation.md) and the in-tree README pointers in
  `onnxruntime/core/providers/js/` and `js/web/lib/wasm/jsep/`, so JSEP contributions can be redirected
  immediately. This is deliberately first: it costs nothing, changes no behavior, and stops new JSEP work
  accumulating while the rest of the plan is executed.
- **Resolve the build-configuration question** (§7), which gates the flip.
- **Close the parity and coverage gaps below.** Items 1–3 need a real browser + GPU/WebNN run; items 4–6 are known
  wiring and coverage gaps to fix.

1. **Proxy-worker (`wasm.proxy = true`).** `./webgpu` already ships this exact path (native WebGPU EP + Asyncify + proxy
   over the EP-agnostic `proxy-wrapper.ts`), so this is a coverage check, not new wiring. CI today runs `--wasm.proxy`
   only with `-b=wasm` (CPU) on the JSEP build — never `-b=webgpu`, and the `--webgpu-ep` leg runs no proxy. Confirm a
   real proxy-mode run on a GPU: native WebGPU EP device init and threading inside a Worker.
2. **IO-binding.** The `'gpu-buffer'` / `'ml-tensor'` paths in `wasm-core-impl.ts` have symmetric native/JSEP
   hooks. CI today runs the `--io-binding=gpu-tensor` / `gpu-location` legs on the JSEP build only; the `--webgpu-ep`
   leg passes no `--io-binding`. Confirm the native path delivers true zero-copy handoff and matching dispose/lifetime
   semantics on a GPU.
3. **WebNN.** WebNN is already the native C++ WebNN EP in both builds (`session-options.ts` selects `WEBNN`
   either way); the flip only swaps the JS init bridge (`jsepInit('webnn', …)` → `webnnInit(…)`, same
   `WebNNBackend`). Confirm the native `webnnInit` path behaves the same in a `navigator.ml`-capable environment.
4. **Global `env.webgpu.*` settings (wiring gap).** `wasm-core-impl.ts` requests a `GPUAdapter` on the JS side and
   then calls `webgpuInit()` without forwarding it, so `adapter`, `powerPreference` and `forceFallbackAdapter` are
   silently dropped on the native path. Resolution:
   - **`powerPreference`** — forward it to the existing `kPowerPreference` EP option
     (`ep.webgpuexecutionprovider.powerPreference` in `webgpu_provider_options.h`), set in `session-options.ts`
     alongside the other `webgpu` options. Note that the two differ when it is *unset*: JSEP passes `undefined`
     to `requestAdapter()` and lets the browser choose, while `WebGpuContextConfig::power_preference`
     (`webgpu_context.h`) defaults to `WGPUPowerPreference_HighPerformance`. Left alone, the flip would move
     users who never expressed a preference onto a discrete GPU, so the wiring needs a behavior-preserving
     default.
   - **`adapter` / `forceFallbackAdapter`** — both are `@deprecated` in `js/common/lib/env.ts`, which names
     `env.webgpu.device` as the replacement. That pointer is wrong: `env.webgpu.device` is output-only — both
     paths write it after init and neither reads it. Custom devices go through the per-session option instead,
     `executionProviders: [{ name: 'webgpu', device }]`, which `session-options.ts` already registers via
     `webgpuRegisterDevice`. Document `adapter` / `forceFallbackAdapter` as no-ops on the native path, and fix
     the `env.webgpu.device` doc comment.
   - Drop the now-dead `navigator.gpu.requestAdapter()` call on the native path once nothing consumes its result.
5. **Profiling.** `env.webgpu.profiling.ondata` is a JSEP-only mechanism: a per-dispatch JavaScript callback
   receiving `kernelId` / `kernelType` / `kernelName` / `programName`, timestamps and input/output tensor metadata
   (`WebGpuProfilingDataV1` in `js/common/lib/env.ts`), falling back to `console.log` when unset. The native
   WebGPU EP has no equivalent hook. It has a richer ORT-framework profiler (`webgpu_profiler.{h,cc}`, gated on
   the session's `enableProfiling`), but its JSON trace is unreachable from JavaScript today:
   `session-handler-inference.ts` leaves `startProfiling()` as a TODO on **both** paths, and `endProfiling()` in
   `wasm-core-impl.ts` frees the filename returned by `_OrtEndProfiling` without surfacing it.

   **Decision:** `ondata` is dropped and documented as JSEP-only. Mapping `env.webgpu.profiling.mode` (and the
   deprecated `profilingMode`) onto the session's `enableProfiling`, implementing `startProfiling()`, and
   surfacing the JSON trace are a **Phase 2 prerequisite** (§10.1), not a flip gate — the `/jsep` escape hatch
   preserves today's behavior for the whole deprecation window.
6. **CI coverage gaps.** Beyond items 1–3, the pipelines themselves need work:
   - `tools/ci_build/github/azure-pipelines/templates/win-wasm-ci.yml` builds only the JSEP variants
     (`wasm_simd_jsep`, `wasm_simd_threads_jsep`) and has no `BuildWebGPU` parameter, so once JSEP is gone the
     Windows/ADO side produces no GPU-capable WASM at all.
   - No CI leg passes `--jspi`, so the `/jspi` bundle is built but never exercised.
   - GitHub Actions leg 07 runs `suite1` while the ADO leg 07 runs the default `suite0` — two nominally identical
     legs testing different things.
   - `--webgpu-ep` rebuilds `dist/ort.all[.min].js` in place and `karma.conf.js` always loads that path, so
     consecutive legs overwrite each other's bundle.

Per-session typed options are already a superset of JSEP's, confirmed by source audit.

---

## 9. Phase 1 — Flip + deprecate (a future release)

1. **Rename the build flag.** `USE_WEBGPU_EP` / `--webgpu-ep` inverts in meaning once the native EP is the
   default. Replace it with a `--jsep` opt-in (`--no-jsep` selecting native). Land the rename while JSEP is still
   the default so it carries no behavior change.
2. **Escape hatch.** Add the temporary, deprecated `onnxruntime-web/jsep` export (built with JSEP selected).
3. **Warn once.** The `/jsep` build warns once (respecting `env.logLevel`) via a shared deprecation-warning
   utility, also used by the WebGL effort; the native default emits nothing.
4. **Retarget the operator-doc generator.** `generate-webgpu-operator-md.ts` parses the JSEP registrations
   (`js_execution_provider.cc`, `js_contrib_kernels.cc`), so post-flip `webgpu-operators.md` describes the wrong
   EP; point it at the native WebGPU EP registrations (those JSEP sources are removed in Phase 2).
5. **Flip the default.** Build `.` and `./all` with the native WebGPU EP. The `webgpu` key and public API are
   unchanged. `/all` keeps WebGL until the WebGL effort removes it. With steps 1–4 already landed this is a
   one-line default change — keep it an isolated, independently revertible commit.
6. **Open the tracking issue.** The public early-warning and parity-report channel: what changed, how to pin
   `/jsep`, and where to report native-WebGPU-EP gaps. Link it from the deprecation notice and the release notes.
7. **Docs.** Deprecation banners + migration guidance in `js/web/README` (hand-authored) and a release-notes entry
   covering the `.jsep.wasm` → `.asyncify.wasm` filename change, `/jsep` pinning, and the int64 `extra` opt-in.
   This is the first user-facing announcement; capture the JSEP usage baseline (§11) before it ships.

**Release gate.** The flip is gated on both: (a) a **blocking** CI job running the native WebGPU EP — the current
native-WebGPU-EP legs are advisory (`continue-on-error` in `windows-web-ci-workflow.yml`, `continueOnError` in
`win-web-ci.yml`) and must be promoted to blocking. The default `.` bundle and `./webgpu` are the same build, so a
green `/webgpu` run covers the flip only if it exercises proxy/IO-binding/WebNN — which the current op-parity leg
does not; and (b) the §8 items resolved with targeted coverage (op-parity tests don't exercise those paths).

---

## 10. Phase 2 — Removal (a subsequent release after Phase 1)

The `/jsep` hatch stays available for **at least** one release. The window is not fixed in advance: it closes at an
explicit checkpoint that reviews parity regressions reported against the tracking issue, the warn-once funnel, and
the JSEP usage signal (per-file CDN statistics for `.jsep.wasm` versus `.asyncify.wasm` and `.jspi.wasm`, which is
the only breakdown npm download totals cannot give). Nothing in the code or the published package hard-codes a
removal release, so extending the window costs only keeping the JSEP CI legs alive.

### 10.1 Prerequisites

Both are non-destructive and must land before any deletion, so that the deletion diffs stay reviewable:

- **Profiling.** Map `env.webgpu.profiling.mode` (and the deprecated `profilingMode`) onto the session's
  `enableProfiling`, implement `startProfiling()`, and stop discarding the `_OrtEndProfiling` trace filename in
  `endProfiling()`. Without this, removing JSEP removes the only working WebGPU profiling in the package (§8.5).
- **Relocate WebNN** out of `js/web/lib/wasm/jsep/` (`backend-webnn.ts`, `webnn/`) to a neutral path, repointing
  importers including `test/test-runner.ts` and `test/unittests/pool-output-shape.ts`.

### 10.2 Web package

1. Drop JSEP WASM artifacts; update `build.ts` / `package.json`; remove the `--jsep` build flag. Repoint `/all`
   to the webgpu/default artifact — only once WebGL has also been dropped (WebGL doc §8); otherwise `/all` stays a
   distinct bundle until then.
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
| Default bundle silently narrows its operator/type surface (reduced-size build args) | High if unaddressed | Measure builds A/B/C and resolve before the flip (§7) |
| int64 behavior change vs. JSEP | Low | None by default (native-off matches JSEP); `enableInt64 = 1` is an opt-in tradeoff |
| Proxy / IO-binding / WebNN unexercised in the native-WebGPU build in CI | Unknown | Validate before flip (§8) |
| Global `env.webgpu.*` settings dropped on native | Medium | `powerPreference` wired with a behavior-preserving default when unset; `adapter` / `forceFallbackAdapter` documented as no-ops, custom devices directed to the per-session `device` option — release gate (§8.4, §9) |
| `env.webgpu.profiling.ondata` has no native equivalent | Low | Documented as JSEP-only; native profiling wiring is a Phase 2 prerequisite (§8.5, §10.1) |
| Windows/ADO produces no WebGPU-EP WASM build | Medium | Add a `BuildWebGPU` leg to `win-wasm-ci.yml` before Phase 2 (§8.6) |
| `post-webnn.js` linkage changes for WebNN once JSEP is removed | Medium | WebNN smoke test on the Phase 2 build, not just a compile check (§10.3) |
| WASM filename change breaks `wasmPaths` | Medium | Document `.jsep.wasm` → `.asyncify.wasm` |
| No telemetry on JSEP adoption | Medium | Per-file CDN statistics for `.jsep.wasm` vs `.asyncify.wasm` / `.jspi.wasm` (baseline captured before the user-facing announcement), plus the warn-once funnel and the escape hatch |

---

## 12. Migration guide

- **Default import (`onnxruntime-web`) and `onnxruntime-web/all`:** no source change if you use default,
  package-managed asset resolution; the `webgpu` backend keeps working and converges to the native WebGPU EP.
  Consumers that pin WASM artifacts must still update the path (see `wasmPaths` below).
- **Lower-overhead build (`onnxruntime-web/jspi`):** on JSPI-capable browsers, prefer `./jspi` — the same native
  WebGPU EP with a smaller WASM binary and lower per-call overhead than Asyncify (the universal fallback).
- **Need JSEP for now:** import `onnxruntime-web/jsep` (temporary, deprecated). File an issue if the
  native WebGPU EP does not work for your model — those reports are what set the removal timeline.
- **WebGPU profiling:** `env.webgpu.profiling.ondata` is JSEP-only and has no native equivalent. Code that
  registers an `ondata` callback keeps working on `/jsep` but silently receives nothing on the default bundle
  (§8.5).
- **Operator and type coverage:** the default bundle's backing artifact changes, and with it the operator/type
  surface available to *all* backends in that bundle, including `wasm` (CPU) — see §7 for the resolution and the
  exact set involved.
- **int64-heavy models:** no change needed — the default matches JSEP. Exceptions that move int64 arithmetic to the
  GPU (lossy for genuine `> 2³¹` values): `extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }`, and
  `enableGraphCapture = true` (forces int64 on; see §6).
- **`wasmPaths` / pinned artifacts:** the backing artifact changes (`.jsep.wasm` → `.asyncify.wasm`). Update
  object-form `env.wasm.wasmPaths`, direct artifact imports, preload/CSP rules, and any copied assets.

> A pinned tracking issue (link TBD) is the authoritative, continuously-updated migration guidance. Warnings,
> docs, and CHANGELOG entries link to it.
