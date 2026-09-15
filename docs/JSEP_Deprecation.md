# JSEP deprecation

**Status: deprecated.** JSEP — the JavaScript/TypeScript WebGPU execution path in `onnxruntime-web` — is being
replaced by the native WebGPU execution provider. Removal is planned. **The timeline is not yet fixed**; it will be
set once we understand real-world JSEP usage, and announced before anything is removed.

This page is the authoritative statement of the deprecation and of the contribution policy below. Link to it when
redirecting JSEP work.

## Contribution policy

JSEP is in maintenance mode: **bug fixes and security fixes only.**

| Change | Where it goes |
|---|---|
| Correctness or security fix in an existing JSEP kernel | JSEP — accepted |
| New operator | The native WebGPU EP — `onnxruntime/core/providers/webgpu/` or `onnxruntime/contrib_ops/webgpu/` |
| New feature, or performance work | The native WebGPU EP |

Note that [`js/web/docs/webgpu-operators.md`](../js/web/docs/webgpu-operators.md) lists **JSEP** operators despite
its name, so it cannot be used to check what the native WebGPU EP already covers.

If a model works on JSEP but not on the native WebGPU EP, that is a gap worth reporting — please open an issue
describing the model and the failure or add support in the native WebGPU EP.

Adding to JSEP now means the work is deleted later and has to be written a second time against the native EP.

## What JSEP is

JSEP implements WebGPU compute in TypeScript, driven from C++ through an Asyncify-compiled WASM core. It spans multiple
areas of the repository:

| Path | Contents |
|---|---|
| `js/web/lib/wasm/jsep/` | The TypeScript WebGPU backend and its kernel implementations |
| `onnxruntime/core/providers/js/` | The native "JS EP" — kernel stubs that dispatch back into JavaScript |
| `onnxruntime/contrib_ops/js/` | Contrib operator registrations for the same EP |
| `onnxruntime/wasm/pre-jsep.js` | Emscripten glue |
| `cmake/onnxruntime_providers_js.cmake`, `js/build_jsep.bat` | Build plumbing |

Built with `--use_jsep` (`USE_JSEP`), and registered under the EP name `JsExecutionProvider`.

This is **not** the same thing as the native WebGPU EP (`onnxruntime/core/providers/webgpu/`), which is a
conventional C++ execution provider (compiled to WASM for onnxruntime-web). Both register under the `webgpu`
backend key in JavaScript. Which one runs is a *build-time* choice, not a runtime one.

## The replacement: native WebGPU EP

`onnxruntime-web/webgpu` and `onnxruntime-web/jspi` are built against the native WebGPU EP **today**. No code
changes and no pending work are required to use them:

```js
import * as ort from 'onnxruntime-web/webgpu';
```

The default `onnxruntime-web` import still selects JSEP; the plan is to change that to the native WebGPU EP. A few
JSEP-only `env.webgpu` settings also have no native equivalent. Both are covered in the
[migration design doc](design/onnxruntime_web_jsep_to_webgpu_ep_migration.md).

## Telling the two apart

Useful when triaging a bug report, since both are called "WebGPU":

| Import | WebGPU implementation | WASM artifact |
|---|---|---|
| `onnxruntime-web` (default) | JSEP | `ort-wasm-simd-threaded.jsep.wasm` |
| `onnxruntime-web/all` | JSEP | `ort-wasm-simd-threaded.jsep.wasm` |
| `onnxruntime-web/webgpu` | native WebGPU EP | `ort-wasm-simd-threaded.asyncify.wasm` |
| `onnxruntime-web/jspi` | native WebGPU EP | `ort-wasm-simd-threaded.jspi.wasm` |

The `.jsep` infix in the WASM filename is the reliable signal. A report that does not identify the import or the
artifact is ambiguous and should be clarified before triage.

## Related documents

- [Migrate onnxruntime-web from JSEP to the native WebGPU EP](design/onnxruntime_web_jsep_to_webgpu_ep_migration.md)
  — the migration design and phasing.
