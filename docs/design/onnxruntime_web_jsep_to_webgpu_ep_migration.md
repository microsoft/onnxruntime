# Design: Migrate onnxruntime-web WebGPU/WebNN from JSEP to the native WebGPU EP

**Status:** Draft
**Last updated:** 2026-07-14
**Scope:** `onnxruntime-web` JavaScript/TypeScript package — WebGPU and WebNN backends

**Related work:** [Remove the WebGL (onnxjs) backend from onnxruntime-web](onnxruntime_web_remove_webgl_backend.md)
— independent, but shares the `onnxruntime-web/all` bundle and the deprecation-warning utility (see §5.3, §8).

---

## 1. Summary

`onnxruntime-web` implements WebGPU/WebNN compute two ways:

- **JSEP** — WebGPU (and WebNN) compute implemented in TypeScript over an Asyncify-compiled WASM core. Powers the
  current default (`.`) and `./all` bundles.
- **Native WebGPU EP** — the C++ WebGPU execution provider compiled to WASM (a port/superset of JSEP). Powers the
  current `./webgpu` and `./jspi` bundles.

This document proposes **deprecating and removing JSEP**, standardizing on the **native WebGPU EP** as the single
WebGPU/WebNN path. The swap is **transparent** for the common case — the default bundle keeps the `webgpu` backend
key and changes implementation at build time — with a **temporary one-release escape hatch** as insurance against
undiscovered parity gaps.

- **Phase 1 (this release):** flip the default (`.`) and `./all` bundles to the native EP, add the temporary
  `onnxruntime-web/jsep` escape-hatch export, and ship deprecation warnings + docs. The flip and the hatch ship
  together, so the native default and its JSEP fallback coexist for exactly one release. The pinned tracking issue
  is published **ahead of** this release as the early-warning channel — no separate warning-only release (§8).
- **Phase 2 (next release):** delete JSEP and remove the temporary `/jsep` export and build flags.

---

## 2. Motivation

- **Duplication of maintenance.** JSEP and the native WebGPU EP implement the same operators twice — once in
  TypeScript and once in C++. The native EP is the strategic direction and receives the bulk of new op work;
  JSEP is effectively a second implementation that must be kept in lockstep.
- **Single, consistent runtime semantics.** Consolidating on the native EP means the browser build shares kernel
  behavior with the rest of ONNX Runtime rather than maintaining a parallel TS implementation.
- **Smaller/simpler build matrix.** Removing JSEP lets us delete the temporary `USE_WEBGPU_EP` / `DISABLE_JSEP`
  build plumbing.

---

## 3. Goals and non-goals

### Goals

- Make the native WebGPU EP the default WebGPU/WebNN backend for `onnxruntime-web`.
- Do so **transparently** for the common consumer (same import, same `webgpu` backend key, same public API).
- Give existing JSEP consumers a clearly-communicated, low-effort migration path plus a one-release safety net.
- Remove the JSEP TypeScript compute path and its build variants.

### Non-goals

- **WebGL removal** — tracked separately in
  [onnxruntime_web_remove_webgl_backend.md](onnxruntime_web_remove_webgl_backend.md).
- **No change to the Node binding** (`onnxruntime-node` / `ort.node.*`).
- **No change to the React-Native package.**
- **No change to ORT training-web.**
- Not introducing new WebGPU features — this is a consolidation, not a feature effort.

---

## 4. Background: how backend selection works today

Backend choice is a **build-time** decision, not a runtime toggle. It is gated by `BUILD_DEFS` flags baked into
each bundle at build time:

- `BUILD_DEFS.DISABLE_JSEP`
- `BUILD_DEFS.DISABLE_WEBGPU` (native EP)
- `BUILD_DEFS.DISABLE_WASM`

The pivot lives in `js/web/script/build.ts` via a temporary `USE_WEBGPU_EP` flag (default `false`). Its
`DEFAULT_DEFINE` sets `DISABLE_JSEP = !!USE_WEBGPU_EP` and `DISABLE_WEBGPU = !USE_WEBGPU_EP`, so a single flag
chooses JSEP vs. the native EP for a given build. Both `USE_WEBGPU_EP` and `USE_JSPI` are already annotated in
the source as temporary and slated for removal.

### 4.1 Current bundle / export map (`js/web/package.json`)

| Export | Artifact | WebGPU/WebNN path |
|---|---|---|
| `.` (default) | `ort.min` / `ort.bundle.min` | JSEP WebGPU + WebNN |
| `./all` | `ort.all.*` | JSEP WebGPU + WebNN (+ WebGL) |
| `./webgpu` | `ort.webgpu.*` | **Native** WebGPU EP + native WebNN |
| `./jspi` | `ort.jspi.*` | Native WebGPU EP + JSPI |
| `./wasm` | `ort.wasm.*` | WASM/CPU only |

The `.` (default) and `./all` bundles are the **JSEP** WebGPU path today. `./webgpu` and `./jspi` are already the
**native** EP. WebNN in the default/`all` bundles is **JSEP-hosted**; in `./webgpu` it is native.

The raw WASM artifacts are variant-specific: `ort-wasm-simd-threaded.jsep.wasm` (JSEP),
`ort-wasm-simd-threaded.asyncify.wasm` (native EP, as used by downstream consumers), `*.jspi.wasm`, and the base
`*.wasm`.

---

## 5. Proposed approach

### 5.1 Transparent default swap (JSEP → native EP)

The default `.` bundle already registers its WebGPU backend under the `webgpu` registry key. The native EP also
registers under `webgpu`. Because selection is a build-time swap and the registry key is identical, flipping the
default bundle from JSEP to the native EP is **transparent**: existing code that requests the `webgpu` EP
continues to work with no source changes.

### 5.2 Escape hatch: temporary `onnxruntime-web/jsep`

After the flip, no existing bundle keeps JSEP (no `/jsep` subpath exists today, and the flip moves both `.` and
`/all` to the native EP). To de-risk, Phase 1 adds a **temporary** `onnxruntime-web/jsep` export (built
`USE_WEBGPU_EP=false`) that pins JSEP for **one release**. It is marked deprecated with a scheduled removal and
emits a one-time warning that doubles as a **parity bug funnel** — surfacing native-EP gaps before JSEP is
deleted.

### 5.3 `/all` bundle: keep as a converging alias

`./all` today differs from the default only by including WebGL. After the WebGL removal (tracked
[separately](onnxruntime_web_remove_webgl_backend.md)) and this JSEP → native swap, it converges to **native
WebGPU EP + native WebNN**, identical to `./webgpu` and the default `.`.

**Decision:** keep `/all` to avoid breaking imports, implemented as a **real alias to the same physical artifact**
as the webgpu/default bundle rather than a separately-built `ort.all.*` — avoiding bundle drift, dropping a build
target, and not doubling the CDN/WASM payload. The backing WASM shifts `.jsep.wasm` → `.asyncify.wasm` with the
general filename migration. "all" becomes a mild misnomer (no longer implies WebGL) — documented, not worth an
export break.

**Sequencing:** the WebGL effort drops WebGL from `/all` first (still JSEP); this swap then converges `/all` onto
the native artifact. Each effort owns its half.

---

## 6. The int64 gap (analysis and decision)

**Mechanics.** The native EP reads int64 support from the session-config entry
`ep.webgpuexecutionprovider.enableInt64` (`webgpu_provider_factory.cc`); absent, it defaults to `false`. The web
layer has no typed field for it, but it is reachable today on `/webgpu` via the untyped `extra` map
(`iterateExtraOptions` flattens `extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }` into the config key).

**Emulation (verified).** When int64 is **enabled** on GPU, JSEP and the native EP use identical **i32-truncated**
emulation (storage `vec2<u32>`, compute `i32`, high word discarded, 32-bit arithmetic). When int64 is **disabled**
(the native default), ops with int64 *inputs* partition to the CPU/WASM fallback using real `int64_t` — full
64-bit correctness. **Exception:** casting *to* int64 stays on GPU even when int64 is off (the WebGPU `Cast`
kernel only guards int64 *inputs*), so "int64 off" narrows to int64 *inputs*, not all int64 work.

**Why it's not a blocker.** int64 in real models carries indices/shapes/axes/token IDs, all `≪ 2³¹` (a tensor
with `> 2³¹` elements can't fit in a `< 2GB` buffer), so JSEP (GPU-truncated) and native-int64-off (CPU-full)
produce identical numeric results for realistic models — the difference is **performance only**. `transformers.js`
already runs the native EP with int64 **off** at scale on int64-heavy token-ID models, confirming the
CPU-fallback cost is acceptable.

**Decision.** Keep int64 **off by default** everywhere, with a uniform native-EP config across `.` and `/webgpu`.
Migrating JSEP users (always-truncating) to native-int64-off is a correctness *upgrade*. Follow-ups (not
blockers): (1) an **optional** typed `enableInt64?: boolean` on `WebGpuExecutionProviderOption` as sugar over the
`extra` key that already works today; (2) benchmark an int64-heavy graph (tokenizer / LLM decode) to quantify the
CPU-fallback cost.

**Graph-capture interaction (resolved).** Graph capture and int64 are coupled inside the EP
(`enable_int64_{config.enable_graph_capture || config.enable_int64}` in `webgpu_execution_provider.cc`): capture
OFF (default) leaves int64 off (int64 ops on CPU, full correctness); capture ON auto-promotes int64 to keep the
captured region all-GPU/capturable. So the `enableInt64` default never breaks graph capture, and capture users
implicitly accept the i32-truncation regime — exactly the LLM-decode token-ID case (`≪ 2³¹`), the status quo
today.

---

## 7. Open investigation items (to resolve during implementation)

Parity concerns to close before the Phase 1 flip. Each is an end-to-end validation: the TypeScript wiring is
confirmable by source audit (noted below), but the runtime behavior needs a real browser + GPU/WebNN check.

1. **Proxy-worker parity (potential blocker).** `wasm.proxy = true` runs compute in a dedicated worker
   (`proxy-wrapper.ts`). *Source:* the wrapper is EP-agnostic and already forbids all GPU I/O over the proxy
   (CPU-I/O-only for both backends). *Runtime:* verify the native EP's device init + threading work inside a
   Worker under Asyncify. (`transformers.js` forces `proxy = false`, so it is not prior art here.)
2. **IO-binding parity (potential blocker).** *Source:* the `'gpu-buffer'` / `'ml-tensor'` paths in
   `wasm-core-impl.ts` have symmetric native/JSEP hooks selected by `BUILD_DEFS.DISABLE_WEBGPU`
   (`webgpuRegisterBuffer`/`jsepRegisterBuffer`, etc.). *Runtime:* confirm the native WASM exports deliver true
   zero-copy handoff and identical dispose/lifetime semantics. Not covered by op-parity tests.
3. **WebNN native-vs-JSEP parity.** The flip also moves default-bundle WebNN from JSEP-hosted to native — a
   second silent swap. *Source:* both init paths are visible in `initEp` (`initJsep('webnn', …)` vs
   `webnnInit(...)`). *Runtime:* requires a `navigator.ml`-capable environment; E2E-only.

**Closed by source audit — typed-option parity.** The native EP branch in `session-options.ts` honors a
**superset** of JSEP's per-session options, so no JSEP-only typed field silently becomes a no-op under native
(JSEP's other configuration comes from the global `env.webgpu.*` object, not per-session options). The only
residual is a source check that those global knobs (profiling in particular) reach the native `webgpuInit` path.

---

## 8. Phase 1 — Flip + deprecate (this release)

Flip the default to the native EP **and** ship the escape hatch, so the swap is protected by a one-release
fallback. Deliverables:

1. **Flip the default.** Build the default `.` and `./all` bundles with the native EP (`USE_WEBGPU_EP=true`). The
   `webgpu` key and public API are unchanged, so the swap is transparent (§5.1). Gated on the release gate below.
2. **Escape hatch.** Add the temporary `onnxruntime-web/jsep` export (built `USE_WEBGPU_EP=false`), deprecated
   with a scheduled removal, shipping in this same release.
3. **JSEP warn-once.** In `wasm-core-impl.ts` `initEp` under `if (!BUILD_DEFS.DISABLE_JSEP)`, for `epName`
   `'webgpu'` (and `'webnn'`). Only the `/jsep` build warns; the native default emits nothing.
4. **Shared deprecation-warning utility.** Warn once and respect `env.logLevel` (error/fatal silences it); no
   separate opt-out flag. Shared with the WebGL-removal effort.
5. **Publish the tracking issue ahead of release.** The pinned migration issue (§11) ships **before** this release
   as the early-warning channel, substituting for a separate warning-only release. Consumers can stay on the
   current pre-flip version until ready, then adopt the flip release and pin the `/jsep` hatch it introduces if
   needed.
6. **int64 `extra` opt-in (documented).** Document `extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }` for
   users needing JSEP-identical GPU int64; works today with no new plumbing, typed field optional (§6).
7. **Docs.** Deprecation banners + migration guidance in `js/web/README` and `js/web/docs/webgpu-operators.md`; a
   prominent release-notes/CHANGELOG entry (WASM filename `.jsep.wasm` → `.asyncify.wasm`, `/jsep` pin
   instructions). Call out that **WebNN** in `.` / `/all` also moves from JSEP-hosted to native WebNN (the same
   path `/webgpu` already ships), so WebNN users know to re-test and use `/jsep` on regression — the
   WebGPU-centric framing otherwise hides this second swap (§7.3).

### Warning placement rationale

The default `.` bundle (native EP) emits nothing — the implementation swap is not actionable for the consumer and
is communicated via the tracking issue and release notes. The `/jsep` escape-hatch bundle warns once ("deprecated
JSEP build, removed in vX; file an issue if the native WebGPU EP does not work for you"), doubling as a parity bug
funnel.

### Release gate

The default flip (#1) is gated on **both**:

1. **Default-bundle suite runs against the native EP.** CI must exercise the default `.` bundle against the native
   EP — a green `/webgpu` run is **not** sufficient, since the bundles differ in wiring (proxy, IO-binding, WebNN
   host) even when op kernels match.
2. **§7 blockers resolved with targeted coverage** (op-parity tests don't exercise these paths): the native EP
   under `wasm.proxy = true` (§7.1); both `'gpu-buffer'` and `'ml-tensor'` IO-binding zero-copy handoff (§7.2);
   native WebNN validated separately (§7.3). Typed-option parity is already closed by source audit; the only
   residual is a source check that the global `env.webgpu.*` knobs reach the native `webgpuInit` path.

Both gates are required — passing #1 while leaving #2 open would ship acknowledged blocker-class parity gaps
unverified.

---

## 9. Phase 2 — Removal (next release)

The default already runs the native EP (flipped in Phase 1); this release deletes JSEP and the temporary surfaces.

1. **Remove build variants:** drop JSEP WASM artifacts; update `build.ts` and `package.json` exports, and remove
   the temporary `USE_WEBGPU_EP` flag. Repoint `/all` to the webgpu/default artifact (§5.3).
2. **Remove guarded code:** delete `BUILD_DEFS.DISABLE_JSEP` and the code it gates; simplify `index.ts`.
3. **Relocate WebNN** out of the `jsep/` directory (`backend-webnn.ts`, `webnn/`) to a neutral path.
4. **Remove `pre-jsep.js` glue;** confirm `post-webgpu.js` / `post-webnn.js` cover all initialization.
5. **Remove the temporary `onnxruntime-web/jsep` export.**

### Removal ergonomics (no tombstones)

The `/jsep` export is removed cleanly, relying on the native bundler error: a lingering
`import 'onnxruntime-web/jsep'` fails with "subpath ./jsep is not defined by exports" — an acceptable build-time
failure on this temporary, opt-in surface. The default `.` import is unaffected (it already swaps to the native
EP), so no consumer on the documented path hits an error. Communicated via release notes and the tracking issue.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Undiscovered native-EP parity gap vs. JSEP | Medium | Temporary `/jsep` escape hatch (1 release) + warn-once bug funnel; differential tests |
| int64 perf regression (CPU fallback) | Low–Med | int64-off is correctness-preserving; benchmark int64-heavy graph; document `extra`/typed opt-in |
| Proxy-worker path behaves differently | Unknown | **Investigate before flip** (§7.1) |
| IO-binding (gpu-buffer/ml-tensor) semantics differ | Unknown | **Investigate before flip** (§7.2) |
| WebNN behavior changes under native path | Unknown | Validate native WebNN separately (§7.3) |
| WASM artifact filename change breaks `wasmPaths` | Medium | Document migration; call out `.jsep.wasm` → `.asyncify.wasm` |
| No telemetry on JSEP adoption | Medium | Warn-once funnel + one-release escape hatch as the feedback mechanism |

---

## 11. Migration guide (for consumers)

- **Default import (`onnxruntime-web`):** no code change needed. The `webgpu` backend continues to work; the
  implementation swaps to the native EP in Phase 1 (this release).
- **`onnxruntime-web/all`:** continues to work and converges to the native EP.
- **Need JSEP for one more release:** import `onnxruntime-web/jsep` (temporary, deprecated). Please file an issue
  if the native WebGPU EP does not work for your model so the gap can be fixed before JSEP is deleted.
- **int64-heavy models needing JSEP-identical GPU behavior:** set
  `extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }` (or the typed option once available). Note this is
  lossy for genuine large-integer int64 data; the default (CPU fallback) is more correct.
- **`wasmPaths`:** the backing WASM artifact changes (`.jsep.wasm` → `.asyncify.wasm`); update any hardcoded
  paths.

> **Canonical source:** this effort has its **own** pinned tracking issue (link TBD) as the authoritative,
> continuously-updated migration guidance, independent of the
> [WebGL-removal](onnxruntime_web_remove_webgl_backend.md) issue. Console warnings, README/docs, and CHANGELOG
> entries **link** to it rather than duplicating guidance. `/all` guidance (touched by both efforts) links to
> whichever issue is relevant.

---

## 12. Resolved decisions

- **Warning suppressibility.** Warn once and respect `env.logLevel` — setting the log level to error/fatal
  silences the deprecation warning; no separate opt-out flag is added. (The WebGL-removal effort adopts the same
  policy.)
