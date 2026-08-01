# Design: Migrate onnxruntime-web from JSEP to the native WebGPU EP

**Scope:** `onnxruntime-web` JavaScript/TypeScript package — WebGPU backend; WebNN initialization glue

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

- **Phase 1 (this release):** flip the default (`.`) and `./all` bundles to the native WebGPU EP, add a temporary
  `onnxruntime-web/jsep` escape-hatch export for one release, and ship deprecation warnings + docs.
- **Phase 2 (subsequent release):** delete JSEP and remove the temporary `/jsep` export and build flags.

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
- Give JSEP consumers a low-effort migration path plus a one-release safety net.
- Remove the JSEP TypeScript compute path and its build variants.

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
- **Escape hatch.** Phase 1 adds a temporary `onnxruntime-web/jsep` export (built `USE_WEBGPU_EP=false`) that pins
  JSEP for one release. It is deprecated and warns once, doubling as a parity-bug funnel.
- **`/all` bundle.** `/all` bundles the WebGPU/WebNN backend and WebGL — two independent things changed by two
  efforts (this one flips WebGPU to native; the WebGL effort drops WebGL). It is kept to avoid breaking
  imports. Once both land, `/all` becomes a real alias of the webgpu/default artifact (`.asyncify.wasm`). The two
  efforts may land in either order; whichever lands second performs the repoint (§9, WebGL doc §8).

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
migration guide (§11).

---

## 7. Items to validate before the flip

Parity checks to close before Phase 1. Items 1–3 need a real browser + GPU/WebNN run; item 4 is a known wiring gap
to fix.

1. **Proxy-worker (`wasm.proxy = true`).** `./webgpu` already ships this exact path (native EP + Asyncify + proxy
   over the EP-agnostic `proxy-wrapper.ts`), so this is a coverage check, not new wiring. CI today runs `--wasm.proxy`
   only with `-b=wasm` (CPU) on the JSEP build — never `-b=webgpu`, and the `--webgpu-ep` leg runs no proxy. Confirm a
   real proxy-mode run on a GPU: native EP device init and threading inside a Worker.
2. **IO-binding.** The `'gpu-buffer'` / `'ml-tensor'` paths in `wasm-core-impl.ts` have symmetric native/JSEP
   hooks. CI today runs the `--io-binding=gpu-tensor` / `gpu-location` legs on the JSEP build only; the `--webgpu-ep`
   leg passes no `--io-binding`. Confirm the native path delivers true zero-copy handoff and matching dispose/lifetime
   semantics on a GPU.
3. **WebNN.** WebNN is already the native C++ WebNN EP in both builds (`session-options.ts` selects `WEBNN`
   either way); the flip only swaps the JS init bridge (`jsepInit('webnn', …)` → `webnnInit(…)`, same
   `WebNNBackend`). Confirm the native `webnnInit` path behaves the same in a `navigator.ml`-capable environment.
4. **Global `env.webgpu.*` settings (wiring gap).** `wasm-core-impl.ts` requests the adapter on the JS side but
   calls `webgpuInit()` without forwarding it to the native WebGPU EP, and `startProfiling()` is a TODO on the native
   path — so `adapter` / `powerPreference` / `forceFallbackAdapter` / `profiling` are silently dropped. Wire these
   into the native path (or document the change) before the flip.

Per-session typed options are already a superset of JSEP's, confirmed by source audit.

---

## 8. Phase 1 — Flip + deprecate (this release)

1. **Flip the default.** Build `.` and `./all` with the native EPs (`USE_WEBGPU_EP=true`). The `webgpu` key and
   public API are unchanged. `/all` keeps WebGL until the WebGL effort removes it.
2. **Escape hatch.** Add the temporary, deprecated `onnxruntime-web/jsep` export (`USE_WEBGPU_EP=false`).
3. **Warn once.** The `/jsep` build warns once (respecting `env.logLevel`) via a shared deprecation-warning
   utility, also used by the WebGL effort; the native default emits nothing.
4. **Publish the tracking issue** ahead of the release as the early-warning channel.
5. **Docs.** Deprecation banners + migration guidance in `js/web/README` (hand-authored) and a release-notes entry
   covering the `.jsep.wasm` → `.asyncify.wasm` filename change, `/jsep` pinning, and the int64 `extra` opt-in.
6. **Retarget the operator-doc generator.** `generate-webgpu-operator-md.ts` parses the JSEP registrations
   (`js_execution_provider.cc`, `js_contrib_kernels.cc`), so post-flip `webgpu-operators.md` describes the wrong
   EP; point it at the native WebGPU EP registrations (those JSEP sources are removed in Phase 2).

**Release gate.** The flip is gated on both: (a) a **blocking** CI job running the native WebGPU EP — the current
native-WebGPU-EP legs are advisory (`continue-on-error` in `windows-web-ci-workflow.yml`, `continueOnError` in
`win-web-ci.yml`) and must be promoted to blocking. The default `.` bundle and `./webgpu` are the same build, so a
green `/webgpu` run covers the flip only if it exercises proxy/IO-binding/WebNN — which the current op-parity leg
does not; and (b) the §7 items resolved with targeted coverage (op-parity tests don't exercise those paths).

---

## 9. Phase 2 — Removal (subsequent release)

Timed one release after Phase 1, keeping the `/jsep` hatch available for exactly one release. Extend only if
native-EP parity regressions surface via real `/jsep` usage.

1. Drop JSEP WASM artifacts; update `build.ts` / `package.json`; remove the `USE_WEBGPU_EP` flag. Repoint `/all`
   to the webgpu/default artifact — only once WebGL has also been dropped (WebGL doc §8); otherwise `/all` stays a
   distinct bundle until then.
2. Delete `BUILD_DEFS.DISABLE_JSEP` and the code it gates; simplify `index.ts`.
3. Relocate WebNN out of `jsep/` (`backend-webnn.ts`, `webnn/`) to a neutral path.
4. Remove `pre-jsep.js` glue; confirm `post-webgpu.js` / `post-webnn.js` cover initialization.
5. Remove the `onnxruntime-web/jsep` export. A lingering `import 'onnxruntime-web/jsep'` then fails with the
   native bundler error — an acceptable build-time failure on this temporary surface. The default `.` import is
   unaffected.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Undiscovered native-EP parity gap vs. JSEP | Medium | One-release `/jsep` escape hatch + warn-once funnel; differential tests |
| int64 behavior change vs. JSEP | Low | None by default (native-off matches JSEP); `enableInt64 = 1` is an opt-in tradeoff |
| Proxy / IO-binding / WebNN unexercised under the native EP in CI | Unknown | Validate before flip (§7) |
| Global `env.webgpu.*` settings dropped on native | Medium | Wire into the native path or document — release gate (§7, §8) |
| WASM filename change breaks `wasmPaths` | Medium | Document `.jsep.wasm` → `.asyncify.wasm` |
| No telemetry on JSEP adoption | Medium | Warn-once funnel + one-release escape hatch |

---

## 11. Migration guide

- **Default import (`onnxruntime-web`) and `onnxruntime-web/all`:** no source change if you use default,
  package-managed asset resolution; the `webgpu` backend keeps working and converges to the native WebGPU EP.
  Consumers that pin WASM artifacts must still update the path (see `wasmPaths` below).
- **Lower-overhead build (`onnxruntime-web/jspi`):** on JSPI-capable browsers, prefer `./jspi` — the same native
  WebGPU EP with a smaller WASM binary and lower per-call overhead than Asyncify (the universal fallback).
- **Need JSEP for one more release:** import `onnxruntime-web/jsep` (temporary, deprecated). File an issue if the
  native EPs do not work for your model.
- **int64-heavy models:** no change needed — the default matches JSEP. Exceptions that move int64 arithmetic to the
  GPU (lossy for genuine `> 2³¹` values): `extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }`, and
  `enableGraphCapture = true` (forces int64 on; see §6).
- **`wasmPaths` / pinned artifacts:** the backing artifact changes (`.jsep.wasm` → `.asyncify.wasm`). Update
  object-form `env.wasm.wasmPaths`, direct artifact imports, preload/CSP rules, and any copied assets.

> A pinned tracking issue (link TBD) is the authoritative, continuously-updated migration guidance. Warnings,
> docs, and CHANGELOG entries link to it.
