# Design: Remove the WebGL (onnxjs) backend from onnxruntime-web

**Scope:** `onnxruntime-web` JavaScript/TypeScript package — WebGL backend

**Related work:**
[Migrate onnxruntime-web from JSEP to the native WebGPU EP](onnxruntime_web_jsep_to_webgpu_ep_migration.md)
— independent, but shares the `onnxruntime-web/all` bundle (§6) and the deprecation-warning utility (§7).

---

## 1. Summary

`onnxruntime-web` ships a legacy **WebGL** backend (the `onnxjs` implementation under `js/web/lib/onnxjs`,
registered under the `'webgl'` key). It predates the WASM architecture, supports fewer operators, and has fp32/op
drift relative to WebGPU.

This document proposes deprecating and then deleting it, with an explicit migration message rather than a
transparent redirect (§5.1):

- **Phase 1 (this release):** deprecation warning + docs. No behavior change.
- **Phase 2 (subsequent release):** delete the `onnxjs` backend, its `'webgl'` registration, and the `ort.webgl`
  build variant.

---

## 2. Motivation

- **Legacy and unmaintained.** The `onnxjs` WebGL backend predates the WASM/EP architecture and does not receive
  new operator or feature work.
- **Narrower op coverage and behavioral drift.** WebGL supports fewer operators than WebGPU and exhibits fp32/op
  differences, making it an inconsistent fallback.
- **Build-matrix simplification.** Removing WebGL deletes the `ort.webgl` bundle variant and the
  `BUILD_DEFS.DISABLE_WEBGL` plumbing, and drops WebGL from the `ort.all` bundle.

---

## 3. Goals and non-goals

### Goals

- Remove the WebGL (`onnxjs`) backend from `onnxruntime-web`.
- Give existing WebGL consumers a clear, actionable migration path (to the WebGPU EP or the WASM/CPU backend).
- Remove the `ort.webgl` build variant and drop WebGL from `ort.all`.

### Non-goals

- **JSEP → native WebGPU EP migration** — tracked separately in
  [onnxruntime_web_jsep_to_webgpu_ep_migration.md](onnxruntime_web_jsep_to_webgpu_ep_migration.md).
- **No change to the Node binding, React-Native package, or ORT training-web.**

---

## 4. Background

The WebGL backend (`onnxjs`, exposed via `backend-onnxjs.ts`) is registered under the `'webgl'` key with
**negative priority**, so it is excluded from the default fallback and used only when explicitly requested
(`executionProviders: ['webgl']`). It appears in two bundles:

| Export | Artifact | Contents |
|---|---|---|
| `./webgl` | `ort.webgl.*` | WebGL only (`DISABLE_WASM: true`) |
| `./all` | `ort.all.*` | JSEP WebGPU + WebNN **+ WebGL** |

The `./webgl` bundle sets `DISABLE_WASM: true`, so it has no WASM/CPU fallback — it is WebGL or nothing.

### 4.1 WebGPU browser coverage

WebGL was historically kept for platforms without WebGPU. As of 2026 that is less true: WebGPU has shipped
enabled-by-default in Chrome/Edge (since **113**, 2023), Safari (macOS/iOS), and Chrome for Android; Firefox
support is partial. Per
[MDN `api.GPU`](https://developer.mozilla.org/docs/Web/API/GPU#browser_compatibility):

| Browser | WebGPU |
|---|---|
| Chrome / Edge (desktop) | ✅ (default since 113; Linux later) |
| Chrome (Android) | ✅ |
| Safari (macOS / iOS) | ✅ |
| Firefox (desktop) | ⚠️ partial |
| Firefox (Android) | ❌ |

Coverage is broadening but not universal, so this design does not assume universal WebGPU — the fallback for
uncovered users is the WASM/CPU backend.

---

## 5. Proposed approach

### 5.1 Explicit removal

WebGL cannot be transparently redirected: the `./webgl` bundle has no WASM/CPU fallback (`DISABLE_WASM: true`), a
silent WebGPU redirect would fail where `navigator.gpu` is unavailable, and WebGL's fp32/op drift would make a
silent swap a hidden behavioral change. So it gets an explicit deprecation warning + removal directing users to
the WebGPU EP (`webgpu`) or the WASM/CPU backend (`wasm`).

### 5.2 Warning placement

The warning fires in `OnnxjsBackend` (`js/web/lib/backend-onnxjs.ts`), in `init(backendName)` /
`createInferenceSessionHandler`. Negative priority means it only fires when WebGL is explicitly requested — for
both `ort.all` and `ort.webgl`.

### 5.3 Removal ergonomics

Phase 2 relies on native failure modes rather than tombstone stubs: a lingering `import 'onnxruntime-web/webgl'`
fails with the bundler error ("subpath ./webgl is not defined by exports"), and `executionProviders: ['webgl']`
throws "no available backend found". Given WebGL's narrow op set and negative priority, these built-in errors are
a sufficient soft landing.

---

## 6. `/all` bundle coupling

`./all` bundles two independent things: WebGL and the WebGPU/WebNN backend. This effort removes WebGL; the
[JSEP → native migration](onnxruntime_web_jsep_to_webgpu_ep_migration.md) flips WebGPU to the native EP. The
two are independent and may land in either order; `/all` collapses into a single alias of the webgpu/default
bundle once both have landed (whichever lands second performs the repoint). `/all` is kept as an export to avoid
breaking imports (see JSEP doc §5).

---

## 7. Phase 1 — Deprecation (this release)

No behavior change.

1. **WebGL warn-once.** In `OnnxjsBackend` (`js/web/lib/backend-onnxjs.ts`), `init(backendName)` /
   `createInferenceSessionHandler`, gated on `logLevel` and deduped via a module flag. Uses the shared
   deprecation-warning utility (from the JSEP effort, or here — whichever lands first).
2. **Docs.** Deprecation banner + migration guidance in `js/web/README` (hand-authored), plus release notes.

---

## 8. Phase 2 — Removal (subsequent release)

Timed one release after Phase 1, kept independent of the JSEP removal schedule. Extend only if WebGL-reliant
consumers report friction migrating to `wasm`/WebGPU.

**Versioning.** ONNX Runtime follows SemVer for its stable public API ([docs/Versioning.md](../Versioning.md)),
but WebGL has always been experimental/maintenance-mode — negative priority, partial op coverage, never GA — so
removing it in a minor release after a deprecation window is within policy and does not require a major bump.

1. Delete the whole `onnxjs` tree, but not with a naive `rm -rf`. Production reaches it only through
   `backend-onnxjs.ts` and the `'webgl'` registration, so remove those plus the WebGL-only code (`onnxjs/backends/webgl`,
   the `test/unittests/backends/webgl/*` tests, and `generate-webgl-operator-md.ts`) outright. First relocate the
   few shared, non-WebGL utilities that test infra imports — protobuf/ONNX decoding (`ort-schema/protobuf`,
   `tensor`), `instrument` (`Logger`/`Profiler`), and `util` (`ProtoUtil`, `PoolConvUtil`, `ShapeUtil`) — to a
   neutral path and repoint their importers; then delete the legacy `onnxjs` session, graph, execution engine, and
   operators (git history preserves them).
2. Remove the `ort.webgl` build variant and the `./webgl` export.
3. Drop WebGL from `ort.all` (§6). If the JSEP → native flip has landed, this collapses `/all` into the
   webgpu/default alias; otherwise `/all` stays a distinct JSEP bundle until it does.
4. Delete `BUILD_DEFS.DISABLE_WEBGL` and the code it gates; simplify `index.ts`.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Consumers relying on WebGL where WebGPU is unavailable | Medium | Coverage is broadening but not universal (§4.1); the fallback is `wasm`; explicit migration message + deprecation window |
| Silent breakage from `./webgl` import removal | Low | Deprecate first; document the removal release; no transparent redirect |
| No telemetry on WebGL adoption | Medium | Warn-once as the feedback mechanism during deprecation |

---

## 10. Migration guide

- **`onnxruntime-web/webgl`:** removed. Migrate to the WebGPU EP (`executionProviders: ['webgpu']`) or the
  WASM/CPU backend (`executionProviders: ['wasm']`). There is no automatic redirect because WebGL builds have no
  WASM fallback and WebGPU requires `navigator.gpu`.
- **`onnxruntime-web/all`:** continues to work, but no longer includes WebGL. If you relied on WebGL as a
  fallback via `/all`, add `wasm` to your `executionProviders` list.

> A pinned tracking issue (link TBD) is the authoritative migration guidance. Warnings, docs, and CHANGELOG
> entries link to it.
