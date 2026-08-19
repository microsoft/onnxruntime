# Can onnxruntime-web load the WebGPU EP as a plugin EP dynamic library?

## Executive summary

**Verdict: technically yes — proven end to end. Economically unproven, and the deciding number
has not been measured. Do the EP-isolation work first, then measure, then decide.**

### What is proven

A working prototype (this directory) runs the real plugin-EP load path under Emscripten dynamic
linking, verified in Node 22 and Chromium 151 across **11 configurations, all passing**:

- **The mechanism works.** `dlopen` → `dlsym("CreateEpFactories")` → calls in both directions
  across the boundary → `ReleaseEpFactory` → `dlclose`, with `ORT_API_VERSION` negotiation and
  C++ exceptions plus RTTI crossing the module boundary correctly. Built against the real ORT
  public headers, so the ABI exercised is the real one.
- **It does not constrain ORT-web's build configuration.** Works with legacy EH *and* native Wasm
  EH, with ASYNCIFY *and* JSPI, with and without pthreads. There is **no browser-version floor** —
  WebAssembly dynamic linking is a toolchain convention, not a browser feature.
- **The host-side size overhead is small.** `-sMAIN_MODULE=2` costs ~6 KB (~21%) on a trivial
  host. (`-sMAIN_MODULE=1` is not viable — ~60x — but is not needed.)
- **An EP can ship its own JS glue.** This was believed to be a hard blocker for WebGPU because
  Dawn's `library_webgpu.js` is linked into the main module. It turns out the host needs only a
  one-line registration hook, not the glue — so onnxruntime-web would not have to carry WebGPU JS
  for users who never load the EP.
- **The one real browser constraint is synchronous loading.** Chromium rejects synchronous
  `WebAssembly.Module` above 8 MB on the main thread, and a real EP is multi-MB. Three working
  alternatives verified: startup preloading, ASYNCIFY, and JSPI.

### What is not known — and it is the deciding number

**How many bytes a real WebGPU side module adds, versus the same EP statically linked.** The
prototype links none of the WebGPU EP, ORT internals or Dawn, so it cannot answer this. For
scale: published `onnxruntime-web` wasm artifacts range 12.9–25.6 MB.

Critically, this must be measured as **isolated-static vs isolated-plugin**, not against today's
build. Much of the duplication is a cost of the *planned EP-isolation effort* — which is paid
whether or not the EP is ever dynamically loaded — and charging it to the plugin decision makes
that decision look worse than it is. See 4.2.

### Recommendation

1. **Do not start with the plugin-EP work.** Land the EP-isolation work first. It is justified on
   its own merits (it also shrinks the native plugin EP DLL), and it is the *enabler*: under
   dynamic linking the side module imports libc/libc++/allocator from the host, so once the EP
   stops linking `framework`/`graph`/`optimizer`/protobuf/ONNX, the plugin path gets cheaper.
2. **Then measure isolated-static vs isolated-plugin.** That single number decides it.
3. **In parallel, if third-party web EPs are the goal, prototype with a small pure-compute EP.**
   It needs neither Dawn nor the glue hook, and shakes out the cmake/CI plumbing at low risk —
   notably the generated dependency list and its CI enforcement (3.3), which is the most
   maintenance-sensitive part of this.
4. **Treat WebGPU as a candidate after that measurement, not before.**

### What would change the answer

- If the isolated-static → isolated-plugin delta is small (say under ~1 MB), WebGPU is a
  perfectly reasonable first customer and the earlier "worst candidate" framing was wrong.
- If it is multi-MB, keep WebGPU statically linked and reserve the plugin-EP path for genuinely
  optional third-party EPs, where the download-only-if-needed benefit actually pays for the
  duplication.

Everything above is backed by the prototype except where explicitly marked as a projection (see
[4.1](#41-what-the-size-cost-actually-is--measured-vs-projected)). See [Reproducing](#reproducing).

---

## 1. What actually blocks it today

| # | Blocker | Where |
|---|---------|-------|
| 1 | The plugin-EP build explicitly refuses Emscripten | `cmake/onnxruntime_providers_webgpu.cmake` — `message(FATAL_ERROR "WebGPU EP shared library build is not supported on Emscripten. Please use static library build.")` |
| 2 | The wasm C API surface has no EP-library registration at all | `onnxruntime/wasm/api.cc` has no `RegisterExecutionProviderLibrary`; `js/web` only knows `OrtAppendExecutionProvider` |
| 3 | The wasm link line is incompatible with dynamic linking | `cmake/onnxruntime_webassembly.cmake`: `-sFILESYSTEM=0`, `-sEXPORT_ALL=0`, no `-sMAIN_MODULE`, nothing built `-fPIC` |
| 4 | The plugin EP statically links a second copy of ORT core | `onnxruntime_providers_webgpu` links `onnxruntime_optimizer`, `onnxruntime_providers`, `onnxruntime_framework`, `onnxruntime_graph`, `onnxruntime_lora`, `onnxruntime_util`, MLAS, `onnx`, protobuf, flatbuffers, abseil. How much survives section GC in a side-module link is **unmeasured** — see 4.1 |
| 5 | Dawn's JS glue is a *main-module* artifact | `emdawnwebgpu_cpp` is linked `PUBLIC` so its `--js-library library_webgpu.js` and `-sASYNCIFY`/`-sJSPI` flags propagate to the final executable. A side module cannot contribute `--js-library` glue at link time — but see 3.5: it *can* be supplied at runtime |

(1)–(3) are mechanical. (4) is the real problem, and 4.1/4.2 argue it is largely not attributable
to the plugin-EP decision. (5) turns out to be surmountable — see 3.5.

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
| `MAIN_MODULE=2`, wasm EH, EP-provided JS glue | 33.3 KB | 2.7 KB | PASS | PASS |
| `MAIN_MODULE=2`, legacy EH, EP-provided JS glue | 31.7 KB | 2.9 KB | PASS | PASS |

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

### 3.5 An EP *can* ship its own JS glue — blocker #5 is surmountable

The strongest WebGPU-specific objection was that `library_webgpu.js` is linked into the final
executable (`emdawnwebgpu_cpp` is `PUBLIC`), and an Emscripten side module cannot contribute
`--js-library` glue at link time. That is true at *link* time, but not at *run* time.

`libdylink.js` resolves a side module's `env` imports against `wasmImports` when the module is
instantiated. Anything already in `wasmImports` at that point resolves — regardless of whether
the host linked it. So the host needs a **hook, not the glue**:

```js
// host_ep_glue_hook.js -- the whole host-side addition
addToLibrary({
  $ortRegisterEpJsGlue: (name, fn) => { wasmImports[name] = fn; },
});
```

`wasmImports` is module-scope and cannot be exposed through `EXPORTED_RUNTIME_METHODS`, but a
JS-library function shares that scope, so exporting this one function is sufficient. The EP then
ships `ep_js_glue.js` next to its `.wasm` and registers before loading. Verified in Node and
Chromium under both exception models: the side module calls a function the host never linked.

Two constraints came out of it:

1. **On-demand loading only.** The glue must be registered before the side module is
   instantiated. `Module.dynamicLibraries` preloading instantiates it inside `createWasm()` —
   after `preInit` (where the exported hook does not exist yet) and before `preRun`. There is no
   usable window, so startup preloading cannot be combined with EP-provided glue. This is not a
   practical problem: ORT registers EPs explicitly via `RegisterExecutionProviderLibrary`, which
   is on-demand by construction.
2. **Glue functions need a signature, and the host needs `-sALLOW_TABLE_GROWTH=1`.** Under legacy
   EH the call is routed through an `invoke_*` trampoline, so the import is *address-taken* and
   becomes a `GOT.func` entry resolved eagerly at load time via `addFunction()`. That requires
   an Emscripten signature annotation (`glue.sig = 'ii'`) and a growable table. Under Wasm EH the
   call is direct and resolves lazily, so it works without either — a difference worth knowing
   about, and not one to rely on.

This does not make the WebGPU EP's Dawn dependency free — the glue still has to be built,
versioned and shipped with the EP, and kept in sync with the Dawn C++ side in the side module.
But it does mean onnxruntime-web would **not** be forced to carry WebGPU JS glue for users who
never load the EP, which was the substance of blocker #5.

---

## 4. Tradeoffs

**Costs**

- **Duplicated ORT core — the dominant cost, and *not measured by this prototype*.** The plugin
  EP target links `onnxruntime_framework`, `graph`, `optimizer`, `providers`, MLAS, ONNX,
  protobuf, flatbuffers and abseil (blocker #4), and `ep/factory.cc` / `ep/ep.cc` still
  `#include "core/framework/..."` and `"core/graph/..."`. Today onnxruntime-web ships *one*
  copy; a plugin split would ship two. See 4.1 for what is and is not known here.
- **Dawn glue.** `library_webgpu.js` is a `--js-library` linked into the final executable today.
  This is **surmountable** (3.5): the EP can ship its own glue and register it at runtime through
  a one-line host hook, so onnxruntime-web need not carry WebGPU JS for users who never load the
  EP. It still has to be built, versioned and shipped with the EP.
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

### 4.2 …but most of that cost is *not* attributable to the plugin-EP decision

There is a separate, already-planned effort to isolate the WebGPU EP source from ORT internals
(the `ep/adapter/` bridge exists precisely for this). That changes the accounting materially,
and cuts in a direction worth being explicit about: **isolation and de-duplication pull in
opposite directions.**

Today the EP calls ORT internals directly and shares one copy inside `onnxruntime.wasm`. An
isolated EP talks only to the EP ABI through the adapter layer, so it carries its own
implementations of facilities ORT core also has. **That hit lands whether the isolated EP is
statically linked or dynamically loaded.** It is a cost of isolation, not of dynamic linking.

So the comparison in 4.1 — today's coupled-static build vs a plugin build — charges the wrong
baseline. The decision-relevant comparison is three-way:

| Build | Status |
|---|---|
| (1) coupled + static | today's shipping build; going away if isolation lands |
| (2) isolated + static | the baseline the plugin decision should be judged against |
| (3) isolated + plugin (dynamic) | what this prototype is about |

**(1) → (2) is the cost of isolation, paid regardless. (2) → (3) is the true cost of the wasm
plugin-EP decision**, and it is much smaller than 4.1's framing implies, because under Emscripten
dynamic linking the side module *imports* libc, libc++ and the allocator from the host rather
than duplicating them (3.3 shows exactly this — `malloc`, `free`, `_Znwm`, the `__cxa_*` family
and RTTI typeinfo all resolve to host exports). What genuinely duplicates in (3) is only what the
EP still links statically that the host also contains — and isolation is precisely the work that
shrinks that set.

Two consequences:

- The prerequisite framing in §5 step 1 ("cut the duplicated core before considering this") is
  better read as "isolation is the enabler". Once the EP no longer links `framework`, `graph`,
  `optimizer`, protobuf and ONNX, the plugin-EP path gets *cheaper*, not more expensive.
- The measurement to run is (2) vs (3), not (1) vs (3). Measuring (1) vs (3) would charge the
  plugin decision for the isolation effort's cost and make it look worse than it is.

---

## 5. What it would take

Roughly in order:

0. **Measure (2) vs (3), not (1) vs (3)** — the isolated-static build against the isolated-plugin
   build (4.2). Charging the plugin decision for the isolation effort's cost makes it look worse
   than it is.
1. **Land the isolation work.** It is the enabler, not merely a prerequisite: once
   `onnxruntime_providers_webgpu` no longer links `framework`, `graph`, `optimizer`, protobuf and
   ONNX, the side module shrinks to EP code + adapter + Dawn, with libc/libc++/allocator imported
   from the host. It is also not web-specific — it shrinks the native plugin EP DLL too.
2. Remove the Emscripten `FATAL_ERROR` and add a `-sSIDE_MODULE=1` + `-fPIC` link path for the
   EP target, with the exception model matched to the host.
3. Build `onnxruntime_webassembly` with `-sMAIN_MODULE=2 -fPIC`, drop `-sFILESYSTEM=0`, and add
   a **generated** dependency list derived from the EP's `env`/`GOT.func`/`GOT.mem` imports,
   routed to the correct mechanism per symbol class (see the table in 3.3 and the working
   derivation in `build.ps1`), with a CI check that fails when the EP grows an import the host
   does not satisfy.
4. Expose `RegisterExecutionProviderLibrary` / `UnregisterExecutionProviderLibrary` in
   `onnxruntime/wasm/api.cc` and a JS binding in `js/web`, as **async** APIs.
5. Add the `ortRegisterEpJsGlue`-style hook plus `-sALLOW_TABLE_GROWTH=1` to the host, and move
   emdawnwebgpu's glue to ship with the EP (3.5).

**Recommendation.** This is more attractive than the first revision of this report concluded,
and the two objections that drove that conclusion have both weakened:

- The Dawn JS glue is **not** immovable (3.5).
- The duplicated-core cost is **mostly not attributable** to this decision if isolation is
  happening anyway (4.2), and isolation actively makes the plugin path cheaper.

The loading mechanism is ready: it works under both exception models, both async mechanisms,
with threads, and with EP-provided JS glue — so it does not constrain ORT-web's existing build
configurations. What is still missing is the one number in 4.1, measured as (2) vs (3).

Sequencing still matters more than the verdict: do the isolation work first because it is
justified on its own merits, then measure. If (2)→(3) is small, WebGPU is a perfectly reasonable
first customer after all. A small pure-compute EP remains the lower-risk way to shake out the
build-system and CI plumbing in parallel, since it needs neither Dawn nor the glue hook.

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
.\build.ps1 -MainModule 2 -WasmEH -EpJsGlue     # EP calls its own runtime-registered JS glue (3.5)
node run.js build-mm2-wasmeh preload            # Node
node browser_run.js build-mm2-wasmeh preload    # headless Chromium
```

Each build directory is deleted and recreated before every build, and `run_all.ps1` skips all
execution for a configuration whose build failed, so a summary row can never report `PASS` from
stale artifacts.

`run.js` / `browser_test.html` modes: `preload` (`Module.dynamicLibraries`), `dlopen`
(synchronous, from the virtual FS), `ondemand` (async `dlopen`, needs ASYNCIFY or JSPI).
`-EpJsGlue` builds require an on-demand mode (`dlopen` / `ondemand`) — see 3.5.

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
| `host_ep_glue_hook.js` | the one-line host hook that lets an EP register its own JS glue (3.5) |
| `ep_js_glue.js` | EP-owned JS glue, shipped with the EP rather than linked into the host (3.5) |
| `legacy_eh_dylink_shim.js` | supplies `getTempRet0`/`setTempRet0` to the side module under legacy EH (3.2) |
| `build.ps1` | build matrix; auto-derives the `MAIN_MODULE=2` dependency list |
| `run.js` | Node driver |
| `browser_test.html` | browser harness + main-thread sync-compile limit probe |
| `browser_run.js` | Playwright/Chromium driver |
| `run_all.ps1` | builds and runs the whole matrix |
