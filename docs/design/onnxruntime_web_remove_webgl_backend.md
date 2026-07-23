# Design: Remove the WebGL (onnxjs) backend from onnxruntime-web

**Scope:** `onnxruntime-web` JavaScript/TypeScript package — WebGL backend

**Related work:**
[Migrate onnxruntime-web WebGPU/WebNN from JSEP to the native WebGPU EP](onnxruntime_web_jsep_to_webgpu_ep_migration.md)
— independent, but shares the `onnxruntime-web/all` bundle and the deprecation-warning utility (see §5–§6).

---

## 1. Summary

`onnxruntime-web` ships a legacy **WebGL** backend — the `onnxjs` implementation under `js/web/lib/onnxjs`,
registered under the `'webgl'` backend key. It predates the current WASM architecture, supports a narrower
operator set, and has fp32/behavioral drift relative to the WebGPU path.

This document proposes **deprecating and then removing** it as a straightforward **deprecate-and-delete** with an
explicit migration message — no transparent redirect (see §5.1), unlike the JSEP → native WebGPU EP migration:

- **Phase 1 (this release):** deprecation warning + docs. No behavior change.
- **Phase 2 (subsequent release):** delete the `onnxjs` backend, its `'webgl'` registration, and the `ort.webgl` build
  variant.

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

The WebGL backend is the `onnxjs` implementation in `js/web/lib/onnxjs`, exposed via `backend-onnxjs.ts` and
registered under the `'webgl'` backend key with **negative priority** — meaning it is excluded from the default
fallback ordering and only used when explicitly requested (`executionProviders: ['webgl']`).

It appears in two bundles:

| Export | Artifact | Contents |
|---|---|---|
| `./webgl` | `ort.webgl.*` | WebGL only (`DISABLE_WASM: true`) |
| `./all` | `ort.all.*` | JSEP WebGPU + WebNN **+ WebGL** |

The `./webgl` bundle sets `BUILD_DEFS.DISABLE_WASM: true`, so it has **no WASM/CPU fallback** — it is WebGL or
nothing.

### 4.1 WebGPU browser coverage

The historical reason to keep WebGL was "WebGPU isn't available on Safari / iOS / Firefox." As of 2026 that is
**less true** — WebGPU has shipped enabled-by-default in Chrome/Edge since **113** (2023), in Safari (macOS/iOS),
and in Chrome for Android; Firefox support is partial. Per MDN's
[`api.GPU`](https://developer.mozilla.org/docs/Web/API/GPU#browser_compatibility) browser-compat data, gaps
remain:

| Browser | WebGPU |
|---|---|
| Chrome / Edge (desktop) | ✅ (default since 113; Linux support landed later) |
| Chrome (Android) | ✅ |
| Safari (macOS / iOS) | ✅ |
| Firefox (desktop) | ⚠️ partial |
| Firefox (Android) | ❌ |

Chrome/Edge and Safari/iOS are solid, but gaps remain (older versions, Firefox for Android, Firefox desktop still
partial). Coverage is **broadening but not universal**, so the pool of users for whom WebGL is the *only* GPU path
is shrinking, not gone. The design does **not** assume universal WebGPU — the actionable fallback for uncovered
users is the WASM/CPU backend.

---

## 5. Proposed approach

### 5.1 Explicit removal

WebGL cannot be transparently redirected (unlike the WebGPU JSEP → native swap):

- The `./webgl` bundle sets `DISABLE_WASM: true`, so there is **no WASM/CPU fallback** to redirect to.
- WebGPU requires `navigator.gpu`; a silent redirect would fail where WebGPU is unavailable.
- WebGL's fp32/op drift means silently swapping results would be a hidden behavioral change.

So WebGL gets an **explicit deprecation warning + removal** directing users to the WebGPU EP (`webgpu`) or the
WASM/CPU backend (`wasm`).

### 5.2 Warning placement

The warning fires in `OnnxjsBackend` (`js/web/lib/backend-onnxjs.ts`), in `init()` /
`createInferenceSessionHandler`. Because the backend has negative priority, it only fires when WebGL is
**explicitly requested** — no spam for consumers who never opt in. Fires for both `ort.all` and `ort.webgl`.

### 5.3 Removal ergonomics

Phase 2 relies on the native failure modes rather than adding tombstone stubs or a `'webgl'`-key shim:

- A lingering `import 'onnxruntime-web/webgl'` fails with the native bundler error ("subpath ./webgl is not
  defined by exports") — a clear build-time failure.
- `executionProviders: ['webgl']` throws "no available backend found" from `resolveBackendAndExecutionProviders`.

WebGL was never a first-class backend (narrow op set, fp32 drift, negative priority), so these built-in errors are
a sufficient soft-landing.

---

## 6. `/all` bundle coupling

`./all` bundles two **independent** things: WebGL **and** the WebGPU/WebNN backend. This effort owns **removing
WebGL from `/all`**; the [JSEP → native migration](onnxruntime_web_jsep_to_webgpu_ep_migration.md) owns flipping
the WebGPU/WebNN half from JSEP to the native EP. The two changes are independent and may land in either order, or
in different releases; `/all` collapses into a single physical alias of the webgpu/default bundle only once
**both** have landed (whichever lands second performs the repoint). `/all` is **kept** as an export to avoid
breaking imports (see JSEP doc §5.3 for the alias decision).

---

## 7. Phase 1 — Deprecation (this release)

No behavior change. Deliverables:

1. **WebGL warn-once.** In `OnnxjsBackend` (`js/web/lib/backend-onnxjs.ts`), `init()` /
   `createInferenceSessionHandler`. Gate on `logLevel`; dedupe via a module flag. Uses the shared
   deprecation-warning utility (built by the JSEP effort, or here — whichever lands first).
2. **Docs.** Deprecation banner + migration guidance in `js/web/README` and `js/web/docs/webgl-operators.md`;
   release notes / CHANGELOG entries.

---

## 8. Phase 2 — Removal (subsequent release)

Timed a **single release** after Phase 1 by default, kept **independent** of the JSEP removal's schedule rather
than forcing the two to align. Extend only if issues surface during deprecation (e.g. WebGL-reliant consumers
report friction migrating to `wasm`/WebGPU) — real-world feedback, not the JSEP timeline, drives any extension.

1. **Delete WebGL:** remove the `onnxjs` backend (`js/web/lib/onnxjs`), the `'webgl'` registration, and
   `backend-onnxjs.ts`.
2. **Remove the `ort.webgl` build variant;** update `build.ts` and `package.json` exports (remove the `./webgl`
   export).
3. **Drop WebGL from `ort.all`** (see §6). If the JSEP → native flip has already landed, this is the step that
   collapses `/all` into the webgpu/default alias; if not, `/all` remains a distinct JSEP WebGPU/WebNN bundle
   until that flip lands.
4. **Remove guarded code:** delete `BUILD_DEFS.DISABLE_WEBGL` and the code it gates; simplify `index.ts`.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Consumers relying on WebGL as a fallback where WebGPU is unavailable | Medium | WebGPU coverage is broadening but still has platform caveats (§4.1); the actionable fallback for uncovered users is `wasm`, not WebGPU; explicit migration message + deprecation window + release notes |
| Silent breakage from `./webgl` import removal | Low | Deprecate first; document removal release; no transparent redirect (avoids hidden drift) |
| No telemetry on WebGL adoption | Medium | Warn-once as the feedback mechanism during the deprecation window |

---

## 10. Migration guide

- **`onnxruntime-web/webgl`:** removed. Migrate to the WebGPU EP (`executionProviders: ['webgpu']`) or the
  WASM/CPU backend (`executionProviders: ['wasm']`). There is no automatic redirect because WebGL builds have no
  WASM fallback and WebGPU requires `navigator.gpu`.
- **`onnxruntime-web/all`:** continues to work, but no longer includes WebGL. If you relied on WebGL as a
  fallback via `/all`, add `wasm` to your `executionProviders` list.

> **Canonical source:** this effort has its **own** pinned tracking issue (link TBD) as the authoritative,
> continuously-updated migration guidance, independent of the
> [JSEP → native migration](onnxruntime_web_jsep_to_webgpu_ep_migration.md) issue. Console warnings, README/docs,
> and CHANGELOG entries **link** to it rather than duplicating guidance. `/all` guidance (touched by both efforts)
> links to whichever issue is relevant.
