# Can onnxruntime-web load the WebGPU EP as a plugin EP dynamic library?

**Short answer: yes, the mechanism works — I have it running end to end in Chromium. But for the
WebGPU EP specifically it is currently a net loss, because the EP side module would duplicate a
large part of ORT core and the Dawn JS glue must stay in the main module anyway.**

Everything below is backed by a working prototype in this directory. See
[Reproducing](#reproducing).

---

## 1. What actually blocks it today

| # | Blocker | Where |
|---|---------|-------|
| 1 | The plugin-EP build explicitly refuses Emscripten | `cmake/onnxruntime_providers_webgpu.cmake` — `message(FATAL_ERROR "WebGPU EP shared library build is not supported on Emscripten. Please use static library build.")` |
| 2 | The wasm C API surface has no EP-library registration at all | `onnxruntime/wasm/api.cc` has no `RegisterExecutionProviderLibrary`; `js/web` only knows `OrtAppendExecutionProvider` |
| 3 | The wasm link line is incompatible with dynamic linking | `cmake/onnxruntime_webassembly.cmake`: `-sFILESYSTEM=0`, `-sEXPORT_ALL=0`, no `-sMAIN_MODULE`, nothing built `-fPIC` |
| 4 | The plugin EP statically links a second copy of ORT core | `onnxruntime_providers_webgpu` links `onnxruntime_optimizer`, `onnxruntime_providers`, `onnxruntime_framework`, `onnxruntime_graph`, `onnxruntime_lora`, `onnxruntime_util`, MLAS, `onnx`, protobuf, flatbuffers, abseil |
| 5 | Dawn's JS glue is a *main-module* artifact | `emdawnwebgpu_cpp` is linked `PUBLIC` so its `--js-library library_webgpu.js` and `-sASYNCIFY`/`-sJSPI` flags propagate to the final executable. A side module cannot contribute `--js-library` glue to the main module. |

(1)–(3) are mechanical. (4) and (5) are the real problems.

---

## 2. The prototype

`host_main.cc` is a mini `onnxruntime.wasm`: it implements a small `OrtApiBase`/`OrtApi` and a
`RegisterExecutionProviderLibrary` that does `dlopen` → `dlsym("CreateEpFactories")` → call →
`ReleaseEpFactory` → `dlclose`, mirroring `EpLibraryPlugin`.

`plugin_ep.cc` is a mini plugin EP side module, structured like
`onnxruntime/core/providers/webgpu/ep/api.cc`.

Both compile against the **real** ORT public headers (`onnxruntime_c_api.h` /
`onnxruntime_ep_c_api.h`), so the ABI exercised is the real one. The prototype verifies:

- `dlopen` / `dlsym` of the EP library, and `dlclose`
- main → side calls through the `OrtEpFactory` vtable
- side → main calls through `OrtApi` function pointers (`CreateStatus`, …)
- `ORT_API_VERSION` negotiation via `OrtApiBase::GetApi()`
- a C++ exception thrown *and* caught inside the side module, with working RTTI
  (`typeid` correctly reports `St13runtime_error` across the boundary)
- an `OrtStatus` allocated by the main module, returned through the side module, freed by the
  main module
- `EM_ASM` from inside the side module (i.e. a side module *can* reach JS)

### Results

Measured on Emscripten 4.0.23, Node 22.16, Chromium 151 (`crossOriginIsolated=true`).

| Config | host .wasm | EP .wasm | Node | Chromium |
|---|---|---|---|---|
| baseline, no dynamic linking | 27.5 KB | – | – | – |
| `MAIN_MODULE=1`, legacy EH | **1623.6 KB** | 2.7 KB | PASS | PASS |
| `MAIN_MODULE=1`, legacy EH, ASYNCIFY | **2351.1 KB** | 5.2 KB | PASS | PASS |
| `MAIN_MODULE=2`, legacy EH | 31.7 KB | 2.7 KB | **FAIL** | – |
| `MAIN_MODULE=2`, wasm EH | 33.3 KB | 2.5 KB | PASS | PASS |
| `MAIN_MODULE=2`, wasm EH, pthreads | 46.8 KB | 2.7 KB | PASS | PASS |
| `MAIN_MODULE=2`, wasm EH, pthreads, JSPI | 46.8 KB | 75.9 KB | – | PASS |

---

## 3. What the prototype taught us

### 3.1 `MAIN_MODULE=1` is not an option — it costs ~60x

`-sMAIN_MODULE=1` disables dead-code elimination and exports everything, so the host grew from
27.5 KB to 1.6 MB (2.3 MB once ASYNCIFY instruments all of it). Applied to the real
`onnxruntime.wasm` this would be a very large regression for every ORT-web user, including
those who never touch WebGPU.

### 3.2 `MAIN_MODULE=2` works, but only with **native Wasm exceptions**

`-sMAIN_MODULE=2` keeps DCE (33.3 KB vs 27.5 KB baseline — about a 6 KB / 21% overhead), but
every symbol the EP imports must be exported by name from the host. With the default
`-fexceptions` (legacy JS-based EH), the side module imports `__THREW__` and `invoke_*`
trampolines which only exist if the *main* module happened to generate them — a side module
cannot add them after the fact. That configuration fails to load.

Switching **both** modules to `-fwasm-exceptions` removes `invoke_*`/`__THREW__` entirely and
everything works. ORT already has a `-fwasm-exceptions` code path.

The exception model must match in both modules — Emscripten links a different libc++/libc++abi
variant per model, and mismatches abort at the first `throw`.

### 3.3 The export list is much more than "functions"

`build.ps1` derives the host's export list mechanically from the side module's `env`,
`GOT.func` and `GOT.mem` imports. For a *trivial* EP that already includes:

```
__cpp_exception (a WebAssembly Tag), __cxa_allocate_exception, __cxa_begin_catch,
__cxa_end_catch, __cxa_free_exception, __cxa_throw, __wasm_lpad_context,
_Unwind_CallPersonality, _ZdlPvm, _Znwm, _ZSt9terminatev,
_ZTISt13runtime_error, _ZTISt9exception,          <-- C++ RTTI *data* symbols
_ZNSt13runtime_errorC1EPKc, _ZNSt13runtime_errorD1Ev,  <-- libc++ inline instantiations
emscripten_builtin_memalign, malloc, free, stderr, pthread_self, printf-family
```

Note `_ZTISt13runtime_error` / `_ZTISt9exception`: these are RTTI **data** symbols, and
`__cpp_exception` is a **WebAssembly tag**. This list is not stable — it changes with compiler
version, optimisation level and EP source changes. Making this robust needs a generated,
CI-verified export list, not a hand-maintained one.

### 3.4 Browsers: dynamic linking itself is fine; **synchronous** loading is not

WebAssembly dynamic linking is a toolchain convention, not a browser feature, so there is no
per-browser support matrix to worry about. What *is* a hard browser constraint is synchronous
compilation. Probing Chromium 151 directly from the harness:

```
sync compile 1MB: OK (1ms)
sync compile 2MB: OK (1ms)
sync compile 4MB: OK (2ms)
sync compile 8MB: REJECTED -> RangeError: WebAssembly.Compile is disallowed on the main
                  thread, if the buffer size is larger than 8MB. Use WebAssembly.compile,
                  compile on a worker thread, or use the flag
                  `--enable-features=WebAssemblyUnlimitedSyncCompilation`.
```

Plain `dlopen()` uses `new WebAssembly.Module(bytes)`. A real WebGPU EP side module would be
several MB, so **synchronous `dlopen` on the main thread is not viable**. Three working
alternatives, all verified in Chromium:

1. `Module.dynamicLibraries = ['...wasm']` — Emscripten loads it asynchronously at startup.
   Works with no ASYNCIFY/JSPI, but it is startup-time, not on-demand.
2. `-sASYNCIFY` — `dlopen()` suspends and fetches asynchronously. The calling export returns a
   Promise, so `RegisterExecutionProviderLibrary` becomes async in JS.
3. `-sJSPI` — same, without whole-program instrumentation, and compatible with
   `-fwasm-exceptions`. `WebAssembly.Suspending` is available in Chromium 151.

Note ASYNCIFY and `-fwasm-exceptions` are **mutually incompatible** (`emcc` warns explicitly).
Since `MAIN_MODULE=2` requires Wasm EH (3.2), the only viable full configuration is
**`MAIN_MODULE=2` + `-fwasm-exceptions` + JSPI**, which pins ORT-web to JSPI-capable browsers
(Chrome/Edge 137+) for the plugin-EP path. Today ORT-web supports ASYNCIFY as the default and
JSPI as an opt-in.

`dynamic linking + pthreads is experimental [-Wexperimental]` is emitted by emcc, but the
threaded configuration passed in both Node and Chromium.

---

## 4. Tradeoffs

**Costs**

- **Duplicated ORT core.** This is the dominant cost. The plugin EP links its own copy of
  `onnxruntime_framework`, `graph`, `optimizer`, `providers`, MLAS, ONNX, protobuf, flatbuffers
  and abseil (blocker #4), and `ep/factory.cc` / `ep/ep.cc` still `#include "core/framework/..."`
  and `"core/graph/..."`. Today onnxruntime-web ships *one* copy; the plugin split would ship
  two. That is a multi-MB regression, in the opposite direction from what a plugin EP is
  supposed to buy you.
- **Dawn glue can't move.** `library_webgpu.js` is a `--js-library` linked into the final
  executable. onnxruntime-web would still have to ship the WebGPU JS glue in the main module
  even when the EP is "optional" — so it never becomes truly WebGPU-agnostic.
- **JSPI-only.** Wasm EH is required for `MAIN_MODULE=2`, and Wasm EH excludes ASYNCIFY, so the
  plugin-EP build is JSPI-only.
- **Runtime cost.** PIC + cross-module calls go through the indirect function table and GOT,
  and nothing inlines across the boundary. The WebGPU EP's hot path is GPU-bound so this is
  probably minor, but it is not free.
- **Brittle export list** (3.3) needing generation + CI enforcement.
- **`-sFILESYSTEM=0` must go**, or EP loading must be routed exclusively through
  `Module.dynamicLibraries` / async `dlopen`.

**Benefits**

- A genuine third-party EP story for the web — the point of the EP ABI. This is the real
  argument, and it does not depend on the WebGPU EP being the one that ships this way.
- Optional download of a large EP for users who don't need it (only pays off once the
  duplication in blocker #4 is fixed).
- One code path: WebGPU EP stops being a special case that is statically linked only for web.

---

## 5. What it would take

Roughly in order:

1. **Cut the duplicated core.** Until `onnxruntime_providers_webgpu` stops statically linking
   ORT internals, a wasm plugin EP is strictly worse on size. This is the prerequisite, and it
   is not web-specific — it also shrinks the native plugin EP DLL.
2. Remove the Emscripten `FATAL_ERROR` and add a `-sSIDE_MODULE=1` + `-fPIC` + `-fwasm-exceptions`
   link path for the EP target.
3. Build `onnxruntime_webassembly` with `-sMAIN_MODULE=2 -fPIC -fwasm-exceptions`, drop
   `-sFILESYSTEM=0`, and add a **generated** `EXPORTED_FUNCTIONS` list derived from the EP's
   `env`/`GOT.func`/`GOT.mem` imports (see `build.ps1` for a working derivation), with a CI
   check that fails when the EP grows an import the host doesn't export.
4. Expose `RegisterExecutionProviderLibrary` / `UnregisterExecutionProviderLibrary` in
   `onnxruntime/wasm/api.cc` and a JS binding in `js/web`, as **async** APIs.
5. Decide where the emdawnwebgpu JS glue lives, since it cannot move into the side module.
6. Make the plugin-EP web build JSPI-only, or keep the static build as the ASYNCIFY fallback.

**Recommendation:** don't chase this for the WebGPU EP right now — it is the worst candidate,
because it is the EP most entangled with the main module's JS glue and it is the one everyone
downloads anyway. Do (1) regardless, since it helps the native plugin EP too. If the goal is
third-party web EPs, a small pure-compute EP is a far better first customer, and this prototype
shows the loading mechanism is ready for one.

---

## Reproducing

Prerequisites, both installed under the worktree root by the steps below:

```powershell
git clone --depth 1 --branch 4.0.23 https://github.com/emscripten-core/emsdk.git ..\emsdk
..\emsdk\emsdk.bat install 4.0.23
..\emsdk\emsdk.bat activate 4.0.23

# headless browser
npm install playwright-core
$env:PLAYWRIGHT_BROWSERS_PATH = "..\pw-browsers"
npx playwright-core install chromium
```

Then:

```powershell
.\run_all.ps1                                   # full matrix, Node + Chromium, prints summary
.\build.ps1 -MainModule 2 -WasmEH               # one config
node run.js build-mm2-wasmeh preload            # Node
node browser_run.js build-mm2-wasmeh preload    # headless Chromium
```

`run.js` / `browser_test.html` modes: `preload` (`Module.dynamicLibraries`), `dlopen`
(synchronous, from the virtual FS), `ondemand` (async `dlopen`, needs ASYNCIFY or JSPI).

### Files

| File | Purpose |
|---|---|
| `host_main.cc` | mini `onnxruntime.wasm` — `dlopen` + `OrtApi` implementation |
| `plugin_ep.cc` | mini plugin EP side module — `CreateEpFactories` / `OrtEpFactory` |
| `build.ps1` | build matrix; auto-derives the `MAIN_MODULE=2` export list |
| `run.js` | Node driver |
| `browser_test.html` | browser harness + main-thread sync-compile limit probe |
| `browser_run.js` | Playwright/Chromium driver |
| `run_all.ps1` | builds and runs the whole matrix |
