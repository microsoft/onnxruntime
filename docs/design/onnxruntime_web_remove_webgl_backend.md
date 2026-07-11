# Design: Remove the WebGL (onnxjs) backend from onnxruntime-web

**Status:** Draft
**Last updated:** 2026-07-10
**Scope:** `onnxruntime-web` JavaScript/TypeScript package — WebGL backend

**Related work:**
[Migrate onnxruntime-web WebGPU/WebNN from JSEP to the native WebGPU EP](onnxruntime_web_jsep_to_webgpu_ep_migration.md).
The two efforts are independent but share the `onnxruntime-web/all` bundle and the deprecation-warning utility;
see §5 and §6 for the coupling.

---

## 1. Summary

`onnxruntime-web` ships a legacy **WebGL** backend — the `onnxjs` implementation under `js/web/lib/onnxjs`,
registered under the `'webgl'` backend key. It predates the current WASM architecture, supports a narrower
operator set, and has known fp32/behavioral drift relative to the WebGPU path.

This document proposes **deprecating and then removing** the WebGL backend. Unlike the JSEP → native WebGPU EP
migration, this is a straightforward **deprecate-and-delete** with an explicit migration message — there is no
transparent redirect (see §5.1).

The work is split into two phases:

- **Phase 1 (this release):** ship a deprecation warning + docs. No behavior change.
- **Phase 2 (next release):** delete the `onnxjs` backend, its `'webgl'` registration, and the `ort.webgl` build
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

### 4.1 WebGPU browser coverage (expanding, but with platform caveats)

A common historical reason to keep WebGL was "WebGPU isn't available on Safari / iOS / Firefox." As of 2026 that
is **less true than it was**, though MDN's `api.GPU` browser-compat data still reports meaningful caveats — the
Chrome and Firefox entries are **partial**, not blanket support:

| Browser | WebGPU (per MDN BCD) |
|---|---|
| Chrome / Edge (desktop) | Partial from 113 (ChromeOS/macOS/Windows); broader/full only in a later version, with Linux limited to newer Intel-gen GPUs |
| Chrome Android | 121+ |
| Safari (macOS) | Full, 26 |
| Safari (iOS) | Full, 26 |
| Chrome / Edge (iOS, via WebKit) | Full, 26 |
| Firefox (desktop) | Partial from 141; **excludes Linux and Intel-based macOS** (Apple-Silicon-first) |
| Firefox for Android | ❌ |

So Safari/iOS coverage is now solid, and Chrome/Firefox are moving in the right direction — but there remain real
gaps: older browser versions, Firefox for Android, Linux (both Chrome's GPU-generation limits and Firefox's
exclusion), and Intel-based Macs on Firefox. WebGPU coverage is **broadening but not yet universal**, so the set
of users for whom WebGL is the *only* GPU path is shrinking rather than gone. This supports removing WebGL over a
deprecation window, but the design does **not** rely on WebGPU being universally available — the actionable
fallback for uncovered users is the WASM/CPU backend, not WebGPU.

---

## 5. Proposed approach

### 5.1 Explicit removal, not a transparent redirect

Unlike the WebGPU JSEP → native swap, WebGL cannot be transparently redirected to another backend:

- The `./webgl` bundle sets `DISABLE_WASM: true`, so there is **no WASM/CPU fallback** to silently fall back to.
- WebGPU requires `navigator.gpu`; a silent redirect would fail on devices without WebGPU support.
- WebGL exhibits fp32/op drift, so silently swapping results would be a hidden behavioral change.

Therefore WebGL gets an **explicit deprecation warning + removal** with an actionable migration message directing
users to the WebGPU EP (`webgpu`) or the WASM/CPU backend (`wasm`).

### 5.2 Warning placement

The warning fires in `OnnxjsBackend` (`js/web/lib/backend-onnxjs.ts`), in `init()` /
`createInferenceSessionHandler`. Because the backend registers with negative priority, this only fires when WebGL
is **explicitly requested**, so it does not spam consumers who never opt into WebGL. It fires for both `ort.all`
and `ort.webgl`.

### 5.3 Removal ergonomics (no tombstones)

When the backend is removed in Phase 2, we do **not** ship throwing-stub "tombstone" modules for the `./webgl`
export or a runtime shim for the `'webgl'` backend key. The rationale:

- Removing the `./webgl` export means a lingering `import 'onnxruntime-web/webgl'` fails with the native bundler
  error ("subpath ./webgl is not defined by exports"). That is a clear, immediate, build-time failure — no shim
  needed.
- Requesting `executionProviders: ['webgl']` alone after removal already throws the generic "no available backend
  found" error from `resolveBackendAndExecutionProviders`.
- WebGL was never a production-grade, first-class backend (narrow op set, fp32 drift, negative priority), so an
  elaborate soft-landing is not warranted.

---

## 6. `/all` bundle coupling

WebGL is one of the two backends in `./all` (the other being JSEP WebGPU/WebNN). Removing WebGL drops it from
`/all`, leaving JSEP (which the
[JSEP → native migration](onnxruntime_web_jsep_to_webgpu_ep_migration.md) subsequently converges to the native
EP). This effort owns **removing WebGL from `/all`**; the JSEP effort owns the **final convergence** of `/all`
onto the native artifact. `/all` itself is **kept** as an export to avoid breaking imports (see the JSEP doc §5.3
for the alias decision).

---

## 7. Phase 1 — Deprecation (this release)

No behavior change. Deliverables:

1. **WebGL warn-once.** In `OnnxjsBackend` (`js/web/lib/backend-onnxjs.ts`), `init()` /
   `createInferenceSessionHandler`. Gate on `logLevel`; dedupe via a module flag. Uses the shared
   deprecation-warning utility (built by the JSEP effort, or here — whichever lands first).
2. **Docs.** Deprecation banner + migration guidance in `js/web/README` and `js/web/docs/webgl-operators.md`;
   release notes / CHANGELOG entries.

---

## 8. Phase 2 — Removal (next release)

1. **Delete WebGL:** remove the `onnxjs` backend (`js/web/lib/onnxjs`), the `'webgl'` registration, and
   `backend-onnxjs.ts`.
2. **Remove the `ort.webgl` build variant;** update `build.ts` and `package.json` exports (remove the `./webgl`
   export).
3. **Drop WebGL from `ort.all`** (see §6).
4. **Remove guarded code:** delete `BUILD_DEFS.DISABLE_WEBGL` and the code it gates; simplify `index.ts`.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Consumers relying on WebGL as a fallback where WebGPU is unavailable | Medium | WebGPU coverage is broadening but still has platform caveats (§4.1); the actionable fallback for uncovered users is `wasm`, not WebGPU; explicit migration message + deprecation window + release notes |
| Silent breakage from `./webgl` import removal | Low | Deprecate first; document removal release; no transparent redirect (avoids hidden drift) |
| No telemetry on WebGL adoption | Medium | Warn-once as the feedback mechanism during the deprecation window |

---

## 10. Migration guide (for consumers)

- **`onnxruntime-web/webgl`:** removed. Migrate to the WebGPU EP (`executionProviders: ['webgpu']`) or the
  WASM/CPU backend (`executionProviders: ['wasm']`). There is no automatic redirect because WebGL builds have no
  WASM fallback and WebGPU requires `navigator.gpu`.
- **`onnxruntime-web/all`:** continues to work, but no longer includes WebGL. If you relied on WebGL as a
  fallback via `/all`, add `wasm` to your `executionProviders` list.

> **Canonical source:** the authoritative, continuously-updated migration guidance for both the WebGL and JSEP
> removals lives in a single pinned tracking issue (link TBD). Console warnings, README/docs, and CHANGELOG
> entries **link** to that issue rather than duplicating this guidance, so there is one place to keep current.

---

## 11. Open questions

1. **Warning suppressibility.** Always-once vs. env opt-out. Recommendation: once + respect `logLevel` so
   error/fatal silences it. (Consistent with the JSEP effort.)
2. **Deprecation window length.** One release (aligned with the JSEP removal) vs. longer. Recommendation: align
   with the JSEP removal for a single coordinated backend-consolidation message.
