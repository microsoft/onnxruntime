# Can onnxruntime-web load the WebGPU EP as a plugin EP dynamic library?

**Short answer: yes, the loading mechanism works — I have it running end to end in Chromium,
under both exception models and both async mechanisms, so it would not constrain ORT-web's
existing build configurations. Whether it is *worth* doing for the WebGPU EP is a separate
question, and it hinges on one quantity this prototype does not measure: how much of ORT core
a real WebGPU side module would duplicate. The Dawn JS glue also has to stay in the main module
regardless.**

Everything below is backed by a working prototype in this directory, except where explicitly
marked as a projection (see [4.1](#41-what-the-size-cost-actually-is--measured-vs-projected)).
See [Reproducing](#reproducing).

---

## 1. What actually blocks it today

| # | Blocker | Where |
|---|---------|-------|
| 1 | The plugin-EP build explicitly refuses Emscripten | `cmake/onnxruntime_providers_webgpu.cmake` — `message(FATAL_ERROR "WebGPU EP shared library build is not supported on Emscripten. Please use static library build.")` |
| 2 | The wasm C API surface has no EP-library registration at all | `onnxruntime/wasm/api.cc` has no `RegisterExecutionProviderLibrary`; `js/web` only knows `OrtAppendExecutionProvider` |
| 3 | The wasm link line is incompatible with dynamic linking | `cmake/onnxruntime_webassembly.cmake`: `-sFILESYSTEM=0`, `-sEXPORT_ALL=0`, no `-sMAIN_MODULE`, nothing built `-fPIC` |
| 4 | The plugin EP statically links a second copy of ORT core | `onnxruntime_providers_webgpu` links `onnxruntime_optimizer`, `onnxruntime_providers`, `onnxruntime_framework`, `onnxruntime_graph`, `onnxruntime_lora`, `onnxruntime_util`, MLAS, `onnx`, protobuf, flatbuffers, abseil. How much survives section GC in a side-module link is **unmeasured** — see 4.1 |
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
| `MAIN_MODULE=2`, legacy EH | 31.7 KB | 2.7 KB | PASS | PASS |
| `MAIN_MODULE=2`, legacy EH, ASYNCIFY | 48.1 KB | 5.2 KB | PASS | PASS |
| `MAIN_MODULE=2`, legacy EH, ASYNCIFY, pthreads | 82.4 KB | 5.4 KB | PASS | PASS |
| `MAIN_MODULE=2`, wasm EH | 33.3 KB | 2.5 KB | PASS | PASS |
| `MAIN_MODULE=2`, wasm EH, pthreads | 46.8 KB | 2.7 KB | PASS | PASS |
| `MAIN_MODULE=2`, wasm EH, pthreads, JSPI | 46.8 KB | 75.9 KB | – | PASS |

Every configuration passes end to end, including the deliberate throw path. `MAIN_MODULE=2`
works with **both** exception models and with **both** ASYNCIFY and JSPI.

---

## 3. What the prototype taught us

### 3.1 `MAIN_MODULE=1` is not an option — it costs ~60x

`-sMAIN_MODULE=1` disables dead-code elimination and exports everything, so the host grew from
27.5 KB to 1.6 MB (2.3 MB once ASYNCIFY instruments all of it). Applied to the real
`onnxruntime.wasm` this would be a very large regression for every ORT-web user, including
those who never touch WebGPU.

### 3.2 `MAIN_MODULE=2` works with **either** exception model

`-sMAIN_MODULE=2` keeps DCE (33.3 KB vs 27.5 KB baseline — about a 6 KB / 21% overhead), but
every symbol the EP imports must be satisfied by the host explicitly.

An earlier revision of this report claimed legacy JS-based EH (`-fexceptions`) could not work
under `MAIN_MODULE=2`, on the grounds that the side module's `invoke_*` trampolines "only exist
if the main module happened to generate them". **That was wrong.** `libdylink.js` synthesises
both `invoke_*` wrappers (`$createInvokeFunction`) and `__cxa_find_matching_catch_*` variants on
demand inside `$resolveGlobalSymbol`, precisely so that side modules can need signatures the
main module never used. The observed failure was caused by this prototype's own dependency
derivation, which filtered those symbols out of the export list.

With a complete derivation, legacy EH works. It needs three things the naive list misses:

1. **`__THREW__` must be exported.** It is a wasm data symbol (`GOT.mem`), not JS-provided.
   Exporting it alone gets registration and factory creation working.
2. **`llvm_eh_typeid_for` must be force-included** via `-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE`,
   since it is a JS-library function the main module itself never references.
3. **`getTempRet0` needs a small custom `--js-library` shim.** This one is a genuine gap:
   `libcore.js` defines the implementation as the JS-only `$getTempRet0`, and the plain
   `getTempRet0` exists only as a `__deps` alias. `libdylink.js` resolves side-module `env`
   imports against `wasmImports`, which JS-only symbols never enter — so
   `DEFAULT_LIBRARY_FUNCS_TO_INCLUDE` cannot fix it. Redefining `getTempRet0`/`setTempRet0` as
   ordinary C-callable library functions (see `legacy_eh_dylink_shim.js`) and forcing their
   emission via `EXPORTED_FUNCTIONS` does. `MAIN_MODULE=1` does not need the shim.

Native Wasm EH (`-fwasm-exceptions`) avoids all three, needing only the `__cpp_exception` tag
export, so it remains the simpler option — but it is a convenience, not a requirement.

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
__THREW__ (legacy EH only)                        <-- wasm data symbol
```

Getting this right needs **three different mechanisms**, and picking the wrong one fails:

| Kind | Mechanism | Examples |
|---|---|---|
| wasm functions, data symbols, tags | `EXPORTED_FUNCTIONS` | RTTI typeinfo, `__THREW__`, `__cpp_exception`, `malloc` |
| JS-library functions | `DEFAULT_LIBRARY_FUNCS_TO_INCLUDE` | `llvm_eh_typeid_for`, `__resumeException` |
| JS-only symbols with no C-callable alias | custom `--js-library` | `getTempRet0`, `setTempRet0` |

Two categories must be left alone: linker/dylink-managed globals (`__memory_base`,
`__stack_pointer`, …), and symbols `libdylink.js` creates on demand (`invoke_*`,
`__cxa_find_matching_catch_*`). One more must be left alone for a subtler reason:
`__asyncify_data` / `__asyncify_state` are created by the ASYNCIFY **Binaryen pass, which runs
after `wasm-ld`** — listing them fails the link with `undefined exported symbol`, and the pass
exports them itself.

This list is not stable: it changes with compiler version, optimisation level, exception model
and EP source. Making it robust needs a generated, CI-verified list, not a hand-maintained one.

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
3. `-sJSPI` — same, without whole-program instrumentation. `WebAssembly.Suspending` and
   `WebAssembly.promising` are available in stock Chromium 151 with no experimental flags.

ASYNCIFY and `-fwasm-exceptions` are mutually incompatible (`emcc` warns explicitly), but since
`MAIN_MODULE=2` works with legacy EH too (3.2), that does **not** force JSPI. Both
`MAIN_MODULE=2` + legacy EH + ASYNCIFY and `MAIN_MODULE=2` + Wasm EH + JSPI pass end to end, so
ORT-web could keep ASYNCIFY as its default and would not be pinned to JSPI-capable browsers.

`dynamic linking + pthreads is experimental [-Wexperimental]` is emitted by emcc, but every
threaded configuration passed in both Node and Chromium.

---

## 4. Tradeoffs

**Costs**

- **Duplicated ORT core — the dominant cost, and *not measured by this prototype*.** The plugin
  EP target links `onnxruntime_framework`, `graph`, `optimizer`, `providers`, MLAS, ONNX,
  protobuf, flatbuffers and abseil (blocker #4), and `ep/factory.cc` / `ep/ep.cc` still
  `#include "core/framework/..."` and `"core/graph/..."`. Today onnxruntime-web ships *one*
  copy; a plugin split would ship two. See 4.1 for what is and is not known here.
- **Dawn glue can't move.** `library_webgpu.js` is a `--js-library` linked into the final
  executable. onnxruntime-web would still have to ship the WebGPU JS glue in the main module
  even when the EP is "optional" — so it never becomes truly WebGPU-agnostic.
- **Runtime cost.** PIC + cross-module calls go through the indirect function table and GOT,
  and nothing inlines across the boundary. The WebGPU EP's hot path is GPU-bound so this is
  probably minor, but it is not free. Not measured.
- **Brittle export list** (3.3) needing generation + CI enforcement across three different
  inclusion mechanisms, plus a custom JS shim if legacy EH is kept.
- **`-sFILESYSTEM=0` must go**, or EP loading must be routed exclusively through
  `Module.dynamicLibraries` / async `dlopen`.

**Benefits**

- A genuine third-party EP story for the web — the point of the EP ABI. This is the real
  argument, and it does not depend on the WebGPU EP being the one that ships this way.
- Optional download of a large EP for users who don't need it (only pays off once the
  duplication in blocker #4 is fixed).
- One code path: WebGPU EP stops being a special case that is statically linked only for web.

**Not a cost, contrary to an earlier revision of this report:** the plugin-EP path does *not*
force JSPI and does not pin ORT-web to a browser floor. Legacy EH + ASYNCIFY works under
`MAIN_MODULE=2` (3.2).

### 4.1 What the size cost actually is — measured vs projected

The prototype side module is only `plugin_ep.cc`. It links **none** of the WebGPU EP, ORT
internals or Dawn, so it says nothing about how many bytes survive archive extraction and
section GC in a real side-module link. The "multi-MB regression" framing is a **projection from
the CMake dependency list, not a result.**

What *can* be measured cheaply is the scale involved. From the published `onnxruntime-web`
1.27.0 npm package (`size-probe/`):

| Artifact | Build | Size |
|---|---|---|
| `ort-wasm-simd-threaded.wasm` | CPU only, full ops | 12.86 MB |
| `ort-wasm-simd-threaded.jspi.wasm` | + WebGPU + WebNN, JSPI, reduced ops | 14.35 MB |
| `ort-wasm-simd-threaded.asyncify.wasm` | + WebGPU + WebNN, ASYNCIFY, reduced ops | 23.13 MB |
| `ort-wasm-simd-threaded.jsep.wasm` | JSEP + WebNN, ASYNCIFY | 25.58 MB |

Two caveats: the WebGPU builds use `reducedSizeBuildArgs` (`--disable_ml_ops`,
`--include_ops_by_config`, …) while the CPU-only build does not, so the first two rows are
**not** directly comparable; and none of these is a side-module build.

The one clean, apples-to-apples comparison is **asyncify vs jspi** — identical
`--use_webgpu --use_webnn` + reduced-ops builds differing only in the async mechanism:
**23.13 MB vs 14.35 MB, so ASYNCIFY costs ~8.8 MB (+61%)**. That corroborates the prototype's
own ASYNCIFY inflation (3.1) at production scale, and it means that if a plugin-EP path were
taken *and* it moved ORT-web to JSPI, the async-mechanism saving could offset a meaningful part
of the duplication cost.

So the honest bound: a duplicated core would add somewhere between "the WebGPU EP's own code"
and "a second full ORT core", i.e. roughly 1.5–13 MB, and **the actual figure is unknown**.
Settling it requires removing the Emscripten `FATAL_ERROR` and doing a real side-module link
with a `--emit-symbol-map` / link-map size breakdown — which is most of the project, not a
quick measurement. That measurement should be step 0 of any serious attempt.

---

## 5. What it would take

Roughly in order:

0. **Measure the duplication.** Everything below is only worth doing if a real WebGPU
   side-module link is not multi-MB larger than today's marginal cost (4.1).
1. **Cut the duplicated core.** Until `onnxruntime_providers_webgpu` stops statically linking
   ORT internals, a wasm plugin EP is likely worse on size. Not web-specific — it also shrinks
   the native plugin EP DLL.
2. Remove the Emscripten `FATAL_ERROR` and add a `-sSIDE_MODULE=1` + `-fPIC` link path for the
   EP target, with the exception model matched to the host.
3. Build `onnxruntime_webassembly` with `-sMAIN_MODULE=2 -fPIC`, drop `-sFILESYSTEM=0`, and add
   a **generated** dependency list derived from the EP's `env`/`GOT.func`/`GOT.mem` imports,
   routed to the correct mechanism per symbol class (see the table in 3.3 and the working
   derivation in `build.ps1`), with a CI check that fails when the EP grows an import the host
   does not satisfy.
4. Expose `RegisterExecutionProviderLibrary` / `UnregisterExecutionProviderLibrary` in
   `onnxruntime/wasm/api.cc` and a JS binding in `js/web`, as **async** APIs.
5. Decide where the emdawnwebgpu JS glue lives, since it cannot move into the side module.

**Recommendation:** don't chase this for the WebGPU EP right now — it is the worst candidate,
because it is the EP most entangled with the main module's JS glue and it is the one everyone
downloads anyway. Do (1) regardless, since it helps the native plugin EP too. If the goal is
third-party web EPs, a small pure-compute EP is a far better first customer, and this prototype
shows the loading mechanism is ready for one — under either exception model and either async
mechanism, so it does not constrain ORT-web's existing build configurations.

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
.\build.ps1 -MainModule 2 -Assertions           # names the next unresolved symbol on failure
node run.js build-mm2-wasmeh preload            # Node
node browser_run.js build-mm2-wasmeh preload    # headless Chromium
```

Each build directory is deleted and recreated before every build, and `run_all.ps1` skips all
execution for a configuration whose build failed, so a summary row can never report `PASS` from
stale artifacts.

`run.js` / `browser_test.html` modes: `preload` (`Module.dynamicLibraries`), `dlopen`
(synchronous, from the virtual FS), `ondemand` (async `dlopen`, needs ASYNCIFY or JSPI).

The published-package sizes in 4.1 come from `..\size-probe`:

```powershell
mkdir ..\size-probe; cd ..\size-probe
npm pack onnxruntime-web
tar -xzf onnxruntime-web-*.tgz
Get-ChildItem -Recurse -Filter *.wasm
```

### Files

| File | Purpose |
|---|---|
| `host_main.cc` | mini `onnxruntime.wasm` — `dlopen` + `OrtApi` implementation |
| `plugin_ep.cc` | mini plugin EP side module — `CreateEpFactories` / `OrtEpFactory` |
| `legacy_eh_dylink_shim.js` | supplies `getTempRet0`/`setTempRet0` to the side module under legacy EH (3.2) |
| `build.ps1` | build matrix; auto-derives the `MAIN_MODULE=2` dependency list |
| `run.js` | Node driver |
| `browser_test.html` | browser harness + main-thread sync-compile limit probe |
| `browser_run.js` | Playwright/Chromium driver |
| `run_all.ps1` | builds and runs the whole matrix |
