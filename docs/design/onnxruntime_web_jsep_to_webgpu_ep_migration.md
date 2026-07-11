# Design: Migrate onnxruntime-web WebGPU/WebNN from JSEP to the native WebGPU EP

**Status:** Draft
**Last updated:** 2026-07-10
**Scope:** `onnxruntime-web` JavaScript/TypeScript package — WebGPU and WebNN backends

**Related work:** [Remove the WebGL (onnxjs) backend from onnxruntime-web](onnxruntime_web_remove_webgl_backend.md).
The two efforts are independent but share the `onnxruntime-web/all` bundle and the deprecation-warning utility; see
§5.3 and §8 for the coupling.

---

## 1. Summary

`onnxruntime-web` implements WebGPU/WebNN compute two ways:

- **JSEP** — WebGPU (and WebNN) compute implemented in TypeScript over an Asyncify-compiled WASM core. This is
  the current default (`.`) and `./all` bundles.
- **Native WebGPU EP** — the C++ WebGPU execution provider compiled to WASM (a port/superset of JSEP). This is
  the current `./webgpu` and `./jspi` bundles.

This document proposes **deprecating and removing JSEP**, standardizing on the **native WebGPU execution
provider** as the single WebGPU/WebNN path. The migration is designed to be **transparent** for the common case
(the default bundle keeps the same `webgpu` backend key and swaps the implementation underneath at build time),
with a **temporary escape hatch** for one release as insurance against undiscovered parity gaps.

The work is split into two phases:

- **Phase 1 (this release):** ship deprecation warnings + docs, plumb the one known config gap (`enableInt64`),
  and add a temporary `onnxruntime-web/jsep` escape-hatch export. No default behavior change.
- **Phase 2 (next release):** flip the default bundle to the native EP, delete JSEP, and clean up the temporary
  build flags and exports.

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

Even with strong parity confidence, flipping the default removes the *only* way for a consumer to keep the JSEP
implementation, since:

- No `/jsep` subpath exists today.
- The only other JSEP-containing bundle is `/all`, which also drags in WebGL.

To de-risk, Phase 1 ships a **temporary** `onnxruntime-web/jsep` export (built with `USE_WEBGPU_EP=false`) that
pins the JSEP implementation for **one release**. It is marked deprecated and removal-scheduled, and it emits a
one-time warning that doubles as a **bug funnel** — surfacing any native-EP parity gaps before JSEP is deleted.

### 5.3 `/all` bundle: keep as a converging alias

`./all` today differs from the default only by including WebGL. After both the WebGL removal (tracked
[separately](onnxruntime_web_remove_webgl_backend.md)) and this JSEP → native swap, its contents converge to
**native WebGPU EP + native WebNN**, i.e. identical to `./webgpu` and the default `.`.

**Decision:** keep `/all` to avoid breaking existing imports. Implement it as a **real alias to the same physical
artifact** as the webgpu/default bundle rather than a separately-built `ort.all.*` that coincidentally matches —
this avoids bundle drift, drops a build target, and prevents doubling the CDN/WASM payload. The backing WASM also
shifts from `.jsep.wasm` to `.asyncify.wasm` as part of the general WASM-filename migration. The name becomes a
mild misnomer ("all" no longer implies WebGL); this is documented but not worth an export break.

**Sequencing:** the WebGL-removal effort drops WebGL from `/all` first (while it is still JSEP); this JSEP → native
swap then converges `/all` onto the native artifact. Each effort owns its half of the `/all` transition.

---

## 6. The int64 gap (analysis and decision)

### 6.1 Mechanics

The native EP reads int64 support as a **session config entry** keyed
`ep.webgpuexecutionprovider.enableInt64` (constant `kEnableInt64` in
`onnxruntime/core/providers/webgpu/webgpu_provider_options.h`), via `config_options.TryGetConfigEntry` in
`webgpu_provider_factory.cc`. Values are the strings `"1"` / `"0"`. When absent, the C++ default is
`enable_int64 = false`.

The web layer never sets it through the typed API: `js/web/lib/wasm/session-options.ts` (`case 'webgpu'`) has no
`enableInt64` handling, and there is no field on `WebGpuExecutionProviderOption`. It is, however, reachable today
on the `/webgpu` bundle via the untyped `extra` map, which `iterateExtraOptions`
(`js/web/lib/wasm/wasm-utils.ts`) flattens into dotted session-config keys:

```js
extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }
```

### 6.2 Emulation semantics (verified)

Both JSEP and the native EP, **when int64 is enabled on GPU**, use identical **i32-truncated** emulation:

- Storage type `vec2<u32>`, value/compute type `i32`.
- Reads use only the low word (`.x`) and discard the high word (`.y`).
- Writes from `i32` sign-extend into `vec2<u32>`.
- All shader arithmetic is 32-bit.

When int64 is **disabled** (the native default), int64 ops partition to the CPU/WASM fallback using real
`int64_t` — full, correct 64-bit behavior.

### 6.3 Fidelity vs. the two failure modes

- **int64 ON (GPU):** maximum JSEP fidelity (byte-identical to JSEP, including its truncation quirks), but lossy
  for genuine large-integer int64 *data* (`|v| ≥ 2³¹`): magnitude truncation, 32-bit overflow wrap, and a value
  type that literally cannot hold an arbitrary 64-bit result.
- **int64 OFF (CPU):** maximum correctness (true 64-bit), at the cost of a CPU round-trip for those ops.

In practice the divergence is **usually moot**: int64 in real models carries indices/shapes/axes/counts/token
IDs, all `≪ 2³¹` (a tensor with `> 2³¹` elements cannot fit in a `< 2GB` buffer). So JSEP (GPU-truncated) and
native-int64-OFF (CPU-full) produce **identical numeric results** for realistic models; the difference is
**performance only**.

**Prior evidence:** `transformers.js` runs the native EP with int64 **off** (Asyncify build, no `extra`) at
scale on int64-heavy token-ID models, confirming the CPU-fallback cost is acceptable in production.

### 6.4 Decision

Keep int64 **off by default** everywhere; use a **uniform native-EP config** across `.` and `/webgpu`. int64 is
**not** a correctness blocker for the swap — the risk is performance, not correctness, and migrating JSEP users
(always-truncating) to native-int64-off is a correctness *upgrade*.

Follow-up actions (not blockers):

1. Optionally expose a typed `enableInt64?: boolean` on `WebGpuExecutionProviderOption` for discoverability, plus
   document the `extra` opt-in.
2. Benchmark an int64-heavy graph (tokenizer / LLM decode) JSEP vs. native-int64-off to quantify CPU-fallback
   cost and confirm the perf trade-off.

### 6.5 Graph-capture interaction (resolved)

Graph capture and int64 are **coupled inside the EP**, so there is no conflict to design around. In
`webgpu_execution_provider.cc`:

```cpp
// enable_int64_ is always true when enable_graph_capture_ is true
enable_int64_{config.enable_graph_capture || config.enable_int64},
```

and the graph-capture kernel registry is built with `RegisterKernels(true, true)` (both flags on). Consequences:

- **Graph capture OFF (default):** int64 stays off → int64 ops on CPU → full 64-bit correctness.
- **Graph capture ON:** the EP **auto-promotes int64 to ON**, keeping int64 ops on-GPU so the captured region is
  all-GPU/capturable.

So leaving `enableInt64` at its default never breaks graph capture — enabling capture flips int64 on
automatically. Graph-capture users implicitly accept the i32-truncation emulation, but that is exactly the
LLM-decode token-ID regime (`≪ 2³¹`) and is already the status quo today.

---

## 7. Open investigation items (to resolve during implementation)

These are potential parity concerns not yet fully verified in source. The first two could be functional/
correctness blockers and should be checked before the Phase 2 default flip.

1. **Proxy-worker parity (potential blocker).** `wasm.proxy = true` spins a dedicated worker
   (`js/web/lib/wasm/proxy-wrapper.ts`). JSEP runs WebGPU compute over Asyncify; the native EP has its own
   threading assumptions. `transformers.js` forces `proxy = false`, so it is **not** prior-art for the proxied
   path. Confirm the native EP behaves identically under `wasm.proxy = true`.
2. **IO-binding parity (potential blocker).** Both `'gpu-buffer'` and `'ml-tensor'` output locations are wired in
   `js/web/lib/wasm/wasm-core-impl.ts`. Verify the native EP provides the same zero-copy GPU-buffer / ML-tensor
   handoff semantics. This will not be caught by op-parity tests.
3. **WebNN native-vs-JSEP parity.** In the default `.` bundle, WebNN is JSEP-hosted; the flip moves those users
   to native WebNN as well — a second silent backend swap under the same default bundle. Validate separately from
   WebGPU.
4. **Typed-option parity.** Audit whether any typed `WebGpuExecutionProviderOption` field is honored by only one
   backend (cache modes, `preferredLayout`, buffer-pool knobs, `forceCpuNodeNames`, etc.). A JSEP-only field
   silently becomes a no-op under the native EP.

---

## 8. Phase 1 — Deprecation (this release)

No default behavior change. Deliverables:

1. **JSEP warn-once.** In `wasm-core-impl.ts` `initEp` inside the `if (!BUILD_DEFS.DISABLE_JSEP)` block, for
   `epName` `'webgpu'` (and `'webnn'`). Only JSEP builds warn; native-EP builds never warn.
2. **Shared deprecation-warning utility.** Once-only, respects `env.logLevel`, with an opt-out consideration.
   Shared with the WebGL-removal effort.
3. **int64 plumbing (prereq).** Plumb `enableInt64` into `js/web/lib/wasm/session-options.ts` (mirroring the
   `enableGraphCapture` handling) so migrating users can opt in if they need JSEP-identical GPU int64.
4. **Escape hatch.** Add the temporary `onnxruntime-web/jsep` export (built `USE_WEBGPU_EP=false`), marked
   deprecated with a scheduled removal release.
5. **Docs.** Deprecation banners + migration guidance in `js/web/README` and `js/web/docs/webgpu-operators.md`;
   release notes / CHANGELOG entries.

### Warning placement rationale

- **Default `.` bundle (still JSEP in Phase 1):** the JSEP warn-once (#1) applies. After the Phase 2 flip it
  becomes native EP and emits nothing — the swap is covered in release notes, not a runtime warning, because it
  is not actionable for the consumer.
- **`/jsep` escape-hatch bundle:** warns once ("deprecated JSEP build, removed in vX; file an issue if the native
  WebGPU EP does not work for you"). Doubles as a parity bug funnel.

---

## 9. Phase 2 — Removal (next release)

1. **Flip default:** native WebGPU EP becomes the `webgpu` backend in `ort` / `ort.all`; remove the temporary
   `USE_WEBGPU_EP` flag.
2. **Remove build variants:** drop JSEP WASM artifacts; update `build.ts` and `package.json` exports. Repoint
   `/all` to the webgpu/default artifact (§5.3).
3. **Remove guarded code:** delete `BUILD_DEFS.DISABLE_JSEP` and the code it gates; simplify `index.ts`.
4. **Relocate WebNN** out of the `jsep/` directory (`backend-webnn.ts`, `webnn/`) to a neutral path.
5. **Remove `pre-jsep.js` glue;** confirm `post-webgpu.js` / `post-webnn.js` cover all initialization.
6. **Remove the temporary `onnxruntime-web/jsep` export.**

### Removal ergonomics (no tombstones)

The `/jsep` export is removed **cleanly** — no throwing-stub "tombstone" module and no runtime shim are left
behind. A consumer still importing `onnxruntime-web/jsep` after removal gets the native bundler error ("subpath
./jsep is not defined by exports"), which is an acceptable, immediate build-time failure on this temporary,
one-release, opt-in surface. The default `.` import is unaffected (it silently swaps to the native EP), so no
consumer following the documented path hits an error. The removal is communicated via release notes and the
pinned tracking issue rather than a runtime shim.

### Release gate

Phase 2 (the default flip) is gated on **both** of the following:

1. **Default-bundle suite runs against the native EP.** The web CI test matrix must exercise the default `.`
   bundle against the native WebGPU EP. Today that suite runs JSEP for the `.` bundle; a green `/webgpu` run is
   **not** sufficient to prove the default flip, because the two bundles differ in wiring (proxy, IO-binding,
   WebNN host) even when op kernels match.
2. **Section 7 blockers resolved with targeted coverage.** The potential blockers in §7 must be closed out before
   the flip — not merely tracked. Op-parity tests do **not** exercise these paths, so each needs dedicated
   coverage:
   - **Proxy-worker path (§7.1):** run the native EP under `wasm.proxy = true` in CI (not just the default
     `proxy = false`).
   - **IO-binding (§7.2):** add tests for both `'gpu-buffer'` and `'ml-tensor'` output locations that assert the
     native EP's zero-copy handoff semantics.
   - **Native WebNN (§7.3):** validate the native WebNN path separately from WebGPU, since the default-bundle
     WebNN users are silently moved from JSEP-hosted to native WebNN by the same flip.

   §7.4 (typed-option parity) should be audited in the same pass; a JSEP-only option silently becoming a no-op is
   a quieter regression than the first three but is closed the same way (test or documented removal).

Passing gate #1 while leaving gate #2 open would let the design meet its stated bar with acknowledged
blocker-class parity gaps still unverified; both are required.

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
  implementation swaps to the native EP in the removal release.
- **`onnxruntime-web/all`:** continues to work and converges to the native EP.
- **Need JSEP for one more release:** import `onnxruntime-web/jsep` (temporary, deprecated). Please file an issue
  if the native WebGPU EP does not work for your model so the gap can be fixed before JSEP is deleted.
- **int64-heavy models needing JSEP-identical GPU behavior:** set
  `extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }` (or the typed option once available). Note this is
  lossy for genuine large-integer int64 data; the default (CPU fallback) is more correct.
- **`wasmPaths`:** the backing WASM artifact changes (`.jsep.wasm` → `.asyncify.wasm`); update any hardcoded
  paths.

> **Canonical source:** the authoritative, continuously-updated migration guidance for both the JSEP and WebGL
> removals lives in a single pinned tracking issue (link TBD). Console warnings, README/docs, and CHANGELOG
> entries **link** to that issue rather than duplicating this guidance, so there is one place to keep current.

---

## 12. Open questions

1. **Default-flip timing.** Phase 1 vs. Phase 2. Recommendation: Phase 2 (conservative).
2. **Warning suppressibility.** Always-once vs. env opt-out. Recommendation: once + respect `logLevel` so
   error/fatal silences it.
3. **WebNN-on-JSEP messaging in Phase 1.** Recommendation: yes — note that `ort` / `ort.all` WebNN moves to the
   native path.
4. **int64 typed option in Phase 1.** Recommendation: yes (discoverability), default off.
