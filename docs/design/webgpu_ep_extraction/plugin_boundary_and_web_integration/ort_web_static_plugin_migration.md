# ORT Web Static Plugin Migration

Follow-up design for moving ORT Web onto the statically linked plugin EP path defined in
[Static Plugin EP Registration](static_plugin_ep_registration_design.md). Build and link evidence for the Emscripten
configuration, the remaining migration steps, and the size and performance analysis that constrains them.

## Emscripten and ORT Web

Making the statically linked WebGPU EP reachable through the plugin path is what motivates this design, so the
Emscripten configuration was built before landing the change: `--build_wasm --enable_wasm_simd
--enable_wasm_threads --use_webgpu static_plugin --use_webnn --target onnxruntime_webassembly`. It configures,
compiles and links cleanly, producing `ort-wasm-simd-threaded.asyncify.{mjs,wasm}`.

`static_plugin` is in fact the only plugin kind Emscripten supports — `cmake/onnxruntime_providers_webgpu.cmake`
raises a `FATAL_ERROR` for `shared_lib` there, since there is no runtime library to load.

The WASM WebGPU jobs that produce the published artifacts pass a bare `--use_webgpu`, which defaults to
`static_lib` (`tools/ci_build/build_args.py` uses `const="static_lib"`), so they still build the internal EP.
`.github/workflows/web.yml` additionally runs a **build-only** `--use_webgpu static_plugin` leg in the
`wasm_Release` job. It keeps the Emscripten registration path compiling, but deliberately publishes nothing: its
output file names are identical to those of the published WebGPU build, and the ORT Web test harness selects the
wasm binary by file name, so the two variants cannot ship side by side. Running the ORT Web suites against the
plugin build therefore waits on switching ORT Web over to it, which is separate work with prerequisites that this
change does not address. See [Migration plan](#migration-plan).

A green build is *not* sufficient evidence that the registration is live. `CreateStaticPluginEpLibraries()` in
`onnxruntime/core/session/plugin_ep/ep_static_plugins.cc` is guarded by
`#if defined(USE_WEBGPU) && defined(ORT_WEBGPU_STATIC_PLUGIN)`; if that define does not reach ORT core the function
still compiles, simply returning an empty vector, and the build is just as green while registering nothing. The
check that actually discriminates is at the symbol level — ORT core must hold an *undefined* reference to the
prefixed entry points, resolved by the provider library:

```
llvm-nm libonnxruntime_session.a          | grep WebGpu_   # U WebGpu_CreateEpFactories / U WebGpu_ReleaseEpFactory
llvm-nm libonnxruntime_providers_webgpu.a | grep WebGpu_   # T WebGpu_CreateEpFactories / T WebGpu_ReleaseEpFactory
```

Both were observed on the Emscripten build, which confirms the guard was live, the D3 entry-point prefixing worked,
and the link resolved core's reference against the provider. That check validates build, link and registration
wiring; runtime behaviour was established separately by the ORT Web prototype described under
[Migration plan](#migration-plan).

A practical build note: the WASM build needs `node` on `PATH` — `node_helper.cmake` does a hard `find_program` and
fails configure without it. emsdk bundles a suitable one under `cmake/external/emsdk/node/<version>/bin`.


## Migration plan

The intended end state is that the WebGPU EP is always a plugin EP and `--use_webgpu static_lib` goes away
entirely, ORT Web included. Building the EP into the WASM binary as a static plugin, which this change enables, is
only the first half of that. The second half is *selection*, and two gaps stand between the current state and a
working ORT Web on the plugin path. Neither is visible to the linker, so the symbol-level evidence above does not
speak to either.

Both gaps have since been closed by a throwaway prototype, which ran the full `js/web` WebGPU operator suite in
Edge against a WASM `static_plugin` binary: 2152 tests, all passing. The design below is therefore validated end
to end rather than merely plausible; what remains is productionization, not discovery. The prototype's findings are
recorded inline in the steps that follow.

**ORT Web selects the EP by name, and that name resolves only to the internal EP.**
`js/web/lib/wasm/session-options.ts` maps the `'webgpu'` backend to `epName = 'WebGPU'`, which reaches
`OrtAppendExecutionProvider` in `onnxruntime/wasm/api.cc` and from there
`OrtApis::SessionOptionsAppendExecutionProvider`. That API is a closed table of hardcoded EP names in
`onnxruntime/core/session/provider_registration.cc`, and its `EpID::WebGPU` case is compiled out under
`ORT_USE_EP_API_ADAPTERS`. On a `static_plugin` build the EP therefore registers at `OrtEnv` creation and is then
unselectable from JS: the call returns "WebGPU execution provider is not supported in this build". This is by
design rather than an oversight — the name-based API is the internal-EP mechanism, and plugin EPs are selected
through `RegisterExecutionProviderLibrary` plus `OrtEpDevice`-based selection — but it does mean ORT Web needs new
plumbing rather than a recompile.

**No WebGPU `OrtEpDevice` exists under Emscripten.** Emscripten was neither `WIN32`, `LINUX` nor `APPLE`, so
`cmake/onnxruntime_common.cmake` fell through to `core/platform/device_discovery_default.cc`, which discovers only
a CPU device. `Factory::GetSupportedDevicesImpl` in `onnxruntime/core/providers/webgpu/ep/factory.cc` creates an
`OrtEpDevice` only for devices of type `OrtHardwareDeviceType_GPU`. The one fallback, the virtual GPU device in the
same function, is opt-in through the `allow_virtual_devices` environment configuration entry and is deliberately
registered without allocator info, because it exists to back a device-free compile-only session. So even after
selection plumbing is added, `GetEpDevices` would offer nothing for JS to select.

A workable order:

1. Make WebGPU visible as a device under Emscripten by adding an Emscripten device discovery implementation.
   `cmake/onnxruntime_common.cmake` gains an `Emscripten` arm selecting a new
   `core/platform/emscripten/device_discovery.cc`, which reports the CPU device via `GetCpuDeviceFromCPUIDInfo()`
   plus a GPU `OrtHardwareDevice` when a synchronous `EM_ASM_INT` check finds `navigator.gpu`. No change to
   `Factory::GetSupportedDevicesImpl` is needed: its existing loop filters on `OrtHardwareDeviceType_GPU`, so a
   discovered GPU flows through and picks up both allocator infos unmodified.

   The prototype first tried the other option — having the factory synthesize the device itself through
   `CreateHardwareDevice` — and it worked, but it is the wrong layer. `GetSupportedDevices` documents its `devices`
   argument as "the `OrtHardwareDevice` instances that are available", so that array is the source of truth, and
   `CreateHardwareDevice` is documented for devices with no hardware behind them ("e.g., virtual"). A browser GPU
   is real hardware that ORT's discovery layer merely failed to enumerate, so the producer is the place to fix it.
   Leaving it in the factory would also oblige every future Emscripten plugin EP to repeat the same workaround.

   A capability check is a legitimate form of discovery here. `core/platform/apple/device_discovery.cc` sets the
   precedent: it enumerates nothing, it asserts that one GPU exists on Apple Silicon and hardcodes the vendor id.
   `navigator.gpu` is a stronger signal than that — an actual runtime capability probe, and the same one ORT Web's
   `initEp` uses to decide whether WebGPU is usable. `EM_ASM` from ORT core is likewise established practice
   (`core/graph/model.cc`, `core/framework/external_data_loader.cc`), not just an EP-layer habit.

   The honest limitation is that no adapter information is available. `_OrtInit` creates the `OrtEnv` and therefore
   triggers discovery, and it runs before `navigator.gpu.requestAdapter()` in ORT Web's `initEp`, so `vendor_id`
   and `device_id` are left at 0. That is acceptable: Apple reports no device id either, and browsers mask
   `GPUAdapterInfo.vendor` for fingerprinting resistance regardless. The device is deliberately *not* tagged with
   `kOrtHardwareDevice_MetadataKey_IsVirtual`, which is what keeps it distinct from the virtual device — that path
   is allocator-less by design and cannot back a real session.

   A later change on `main` added a second fallback to `GetSupportedDevicesImpl` alongside the virtual device: when
   no GPU device is discovered, `allow_software_adapter` (opt-in via the `kAllowSoftwareAdapterEnvironmentVariable`
   environment variable) advertises WebGPU against the *CPU* `OrtHardwareDevice`. It does not change the analysis
   above. Both upstream fallbacks work around a device the discovery layer failed to enumerate; they are opt-in
   precisely because misreporting a device is not safe to do by default. Under Emscripten the device is genuinely
   there and cheaply detectable, so fixing discovery keeps the default path correct and needs no opt-in. The
   primary matching loop that our discovered GPU device flows through is unchanged by that work.

   One accepted side effect: `ep_library_internal.cc` creates an internal `WebGpuEpFactory` in a
   `USE_WEBGPU && !ORT_USE_EP_API_ADAPTERS` build, so today's `static_lib` ORT Web build starts producing an
   `OrtEpDevice` where it previously produced none. That build selects by name and is slated for removal, so the
   change is benign.

2. Expose plugin EP selection in the WASM C API. `g_env` is already available in `api.cc`, so `GetEpDevices` and
   `SessionOptionsAppendExecutionProvider_V2` are both reachable. A narrow helper that appends by EP name and
   resolves the device internally avoids marshalling arrays of device pointers across the JS boundary. The new
   entry point needs to be exported from `cmake/onnxruntime_webassembly.cmake`.

   *Prototype result:* the name-resolving helper is the right shape. Note that exporting it takes three edits, not
   one: `EMSCRIPTEN_KEEPALIVE` in `api.h`, the `JSPI_EXPORTS` list in `cmake/onnxruntime_webassembly.cmake` for
   JSPI builds, and the `wrapAsyncAPIs` list in `onnxruntime/wasm/pre-async.js` for Asyncify builds. Omitting the
   last one is the easy mistake, and Asyncify is the default variant. Reporting matters too: on no match the helper
   should raise a status naming the EPs that *are* registered, which is what makes a misconfiguration diagnosable
   from JS through `checkLastError`.

3. Switch `js/web` to the new call. The `BUILD_DEFS.DISABLE_WEBGPU` branch that falls back to the JS EP is
   unaffected. Confirm that the EP options ORT Web passes today — `deviceId`, `webgpuInstance`, `webgpuDevice`,
   `preferredLayout` and the buffer cache modes — still reach the provider, since `Factory::CreateEpImpl` obtains
   them from the session config options.

   *Prototype result:* the options do reach the provider unchanged, and the reason is worth recording. Both paths
   derive their config prefix from `OrtSessionOptions::GetProviderOptionPrefix` keyed on the EP name, so V2 lands
   the options under exactly the same `ep.webgpuexecutionprovider.*` keys the name-based path used. One JS change
   is required though: `session-options.ts` passes the short name `'WebGPU'`, whereas `OrtEpDevice` selection
   matches the canonical `'WebGpuExecutionProvider'` returned by `EpDevice_EpName`.

   The productionization question this raises is *conditionality*: is it safe to switch to V2 unconditionally
   whenever WebGPU is enabled? It is, provided the WASM build flips at the same time. `BUILD_DEFS.DISABLE_WEBGPU`
   is defined in `js/web/script/build.ts` as exactly `!USE_WEBGPU_EP`, so the bundle is already specialized per
   WASM flavour by `--webgpu-ep` — the JSEP bundle takes the other branch entirely and is unaffected. Flipping the
   WebGPU WASM builds to `static_plugin` (step 4) therefore makes `!DISABLE_WEBGPU` imply `static_plugin` by
   construction, and no additional build define or runtime capability probe is needed. What is *not* safe is
   landing the JS change without the build change, since the two must move together.

4. Flip the ORT Web WASM builds to `static_plugin` and extend CI to *test* the result. The two builds are
   `Build (simd + threads + WebGPU experimental)` and its JSPI sibling in
   `.github/workflows/linux-wasm-ci-build-and-test-workflow.yml`, both of which pass a bare `--use_webgpu` and so
   get `static_lib` from `build_args.py`'s `const="static_lib"`. Neither uses `--minimal_build`, so the
   minimal-build `FATAL_ERROR` guard does not stand in the way of this migration. A build-only `static_plugin`
   leg already runs (see [Emscripten and ORT Web](#emscripten-and-ort-web)), so this path is covered at compile
   time; flipping these two builds is what gets it *executed*, because the published artifacts then become the
   plugin build and the existing `web_Debug` / `web_Release` suites cover it with no parallel test lane.

   *Prototype note:* the `js/web` test runner cannot currently exercise this configuration end to end, because
   `script/test-runner-cli.ts` spawns `script/build` with only `--bundle-mode` and does not forward `--webgpu-ep`.
   Setting `npm_config_webgpu_ep` in the environment works as a stopgap, since `script/build.ts` reads it, but a CI
   leg should forward the flag properly. Note also that without `--webgpu-ep` the `webgpu` backend silently routes
   through JSEP instead, so a leg that omits it would pass while testing nothing relevant.

## Interaction with the ORT Web reduced-size build options

The ORT Web WASM builds are size-sensitive and pass a set of size-reduction flags that the local prototype builds
did not. `linux-wasm-ci-build-and-test-workflow.yml` applies `--disable_ml_ops --disable_generation_ops
--disable_types string float4 float8 optional sparsetensor --include_ops_by_config
onnxruntime/wasm/reduced_types.config --enable_reduced_operator_type_support`, and `web.yml` adds `--disable_rtti`
for the release job. Since the migration changes how the WebGPU EP is compiled and linked, each of these was
checked against the plugin path.

**None of them need adapting.** The findings:

- *Compile definitions reach the WebGPU target in every mode.* `onnxruntime_providers_webgpu` is created through
  `onnxruntime_add_static_library` / `onnxruntime_add_shared_library_module`, both of which call
  `onnxruntime_configure_target` → `onnxruntime_set_compile_flags`. That function is where every `DISABLE_*` and
  `REDUCED_OPS_BUILD` definition is applied, so the `static_lib`, `static_plugin` and shared-library builds all
  receive an identical set. There is no propagation gap introduced by the plugin boundary.

- *Operator and type reduction never applied to WebGPU, before or after.*
  `op_registration_utils.get_kernel_registration_files()` hardcodes the CPU registration files (plus CUDA when
  requested); WebGPU is not in the list, so `--include_ops_by_config` does not rewrite WebGPU kernel
  registrations. `onnxruntime/wasm/reduced_types.config` is `!no_ops_specified_means_all_ops_are_required` — it
  performs global *type* reduction only, with no op exclusion — and that mechanism works through the
  `op_kernel_type_control` macros, which neither `core/providers/webgpu` nor `contrib_ops/webgpu` uses. Its saving
  is entirely CPU-EP-side and is unchanged by the plugin switch. This is a pre-existing gap, not a regression.

- *The type and ML/generation op flags are inert for WebGPU.* No source under `core/providers/webgpu` or
  `contrib_ops/webgpu` references `DISABLE_ML_OPS`, `DISABLE_GENERATION_OPS`, `DISABLE_SPARSE_TENSORS`,
  `DISABLE_OPTIONAL_TYPE`, `DISABLE_FLOAT8_TYPES`, `DISABLE_FLOAT4_TYPES` or `DISABLE_STRING_TYPE`.

- *`--disable_rtti` is safe on the plugin path.* Neither `include/onnxruntime/ep` (the EP API adapters) nor
  `onnxruntime/core/session/plugin_ep` uses `dynamic_cast` or `typeid`.

One consequence is worth recording for the *shared library* plugin EP, which is out of scope here but shares this
code. `onnxruntime_c_api.h` contains no `DISABLE_*` guards, so the `OrtApi` struct layout is invariant under these
flags and the ABI is stable. `onnxruntime_cxx_api.h` does guard some `Ort::Value` members (the sparse tensor
methods) — an EP DLL built with a different `--disable_types` than its host would therefore see a different C++
header surface, but since those wrappers are inline over stable function pointers this is a source-compatibility
concern rather than an ABI one. For `static_plugin` everything is compiled in one tree with one set of defines, so
the question does not arise.

**What is *not* covered by any existing flag** is the plugin machinery itself. Relative to `static_lib`, the
`static_plugin` build adds `core/providers/webgpu/ep/*` (excluded from the source list in the `static_lib` branch
of `onnxruntime_providers_webgpu.cmake`), inlines the header-only EP API adapters into every WebGPU translation
unit, and reaches the `OrtGraph` / `OrtNode` C API graph views through `OrtEp::GetCapability`. Note that
`core/session/plugin_ep` is *not* part of that delta: `onnxruntime_session.cmake` only excludes it for
`onnxruntime_MINIMAL_BUILD`, so today's WASM builds already compile it in.

That also means the only existing lever that trims plugin-EP machinery is `--minimal_build`, which
`static_plugin` currently rejects outright. If measurement shows the size delta matters, lifting that restriction
(already listed as a follow-up) is the mechanism to reach parity — not a new size-reduction flag.

### Measured size comparison

The ORT Web npm package ships **two** WebGPU WASM artifacts, not one, and both were measured. The release path
(`npm-packaging-pipeline.yml` → `templates/web-ci.yml` → `templates/linux-wasm-ci.yml`) and the GitHub Actions path
(`web.yml` → `linux-wasm-ci-build-and-test-workflow.yml`) pass an identical flag set for the WebGPU legs, which
differ from each other only in their exception/stack-switching mode:

| Leg | Artifact | Distinguishing flag |
| --- | --- | --- |
| `wasm_inferencing_webgpu` | `ort-wasm-simd-threaded.asyncify.{wasm,mjs}` | `--enable_wasm_api_exception_catching` |
| `wasm_inferencing_webgpu_jspi` | `ort-wasm-simd-threaded.jspi.{wasm,mjs}` | `--enable_wasm_jspi` |

Four clean from-scratch builds were run (both EP modes × both legs) with the exact release flag set: `Release`,
SIMD + threads, WebNN on, `--disable_rtti`, `--target onnxruntime_webassembly`, and the full reduced-size set
above. Compressed sizes are included because that is what is actually delivered over the wire; they were produced
with gzip level 9 and brotli quality 11.

`ort-wasm-simd-threaded.asyncify.wasm`:

| Metric | `static_lib` | `static_plugin` | Delta |
| --- | ---: | ---: | ---: |
| raw | 26,549,423 B | 26,487,397 B | **−62,026 B (−0.234%)** |
| gzip | 6,537,701 B | 6,566,088 B | **+28,387 B (+0.434%)** |
| brotli | 3,953,625 B | 3,971,550 B | **+17,925 B (+0.453%)** |

`ort-wasm-simd-threaded.jspi.wasm`:

| Metric | `static_lib` | `static_plugin` | Delta |
| --- | ---: | ---: | ---: |
| raw | 16,627,577 B | 16,687,113 B | **+59,536 B (+0.358%)** |
| gzip | 4,099,914 B | 4,126,147 B | **+26,233 B (+0.640%)** |
| brotli | 2,651,643 B | 2,674,520 B | **+22,877 B (+0.863%)** |

The accompanying `.mjs` files are byte-identical between the two EP modes in both legs (53,300 B asyncify, 51,204 B
JSPI); their compressed sizes differ by at most 23 B.

**Every delta is under 1%, which is the substantive conclusion.** Two details are worth recording so the numbers
are not over-read:

- *The raw sign is not stable across the two artifacts.* `static_plugin` is marginally smaller raw on asyncify but
  marginally larger raw on JSPI. An earlier revision of this document reported only the asyncify raw number and
  generalised it to "no size regression"; that generalisation was too broad. The asyncify measurement itself
  reproduces well — it was −61,968 B when first taken and −62,026 B in this from-scratch rerun at a later commit.

- *Compressed size grows slightly in both legs*, by +0.4% to +0.9%, so the direction is consistent once the
  artifacts are compressed. Why compressed size can rise while raw size falls was not investigated; no
  symbol-level or per-section attribution was done, and none of the earlier speculation about *which* code the
  plugin build drops or adds has been measured. Given the magnitude, this was not pursued further.

Both configurations compiled cleanly with every reduced-size flag in both legs, which empirically confirms the
compatibility analysis above. For reference, the same `static_plugin` asyncify configuration without any
reduced-size flags is 40.9 MB, so those flags are worth ~15 MB and remain essential regardless of EP mode.

### `--enable_wasm_api_exception_catching` and the EP API adapters

The one option that *did* interact with the plugin path is `--enable_wasm_api_exception_catching`. It is not a
size-reduction flag in the same sense as the others, but it is part of the same CI flag set and it exposed a real
bug.

`onnxruntime_webassembly.cmake` compiles only `wasm/api.cc` and `core/session/onnxruntime_c_api.cc` with
`-sDISABLE_EXCEPTION_CATCHING=0`; every other translation unit keeps Emscripten's default, where `catch` clauses
are compiled so that they never match. Exceptions still propagate — they are simply only catchable at the C API
boundary.

The EP API adapters in `include/onnxruntime/ep/adapter/` are header-only and are therefore inlined into WebGPU
translation units that have catching disabled. `OpKernelInfo::GetAttr()` / `GetAttrs()` were implemented as a
`try` around the throwing `Ort::` C++ wrappers, converting `Ort::Exception` into a `Status`. That makes a *missing
optional attribute* — ordinary control flow for `GetAttrOrDefault()` — depend on catching an exception. With
catching disabled the `catch` never ran, so the exception escaped `GetAttrOrDefault()` entirely and surfaced from
session creation as e.g. `ERROR_CODE: 6, ERROR_MESSAGE: No attribute with name:'extrapolation_value'is defined.`

This reproduced as 949 `suite0` failures on the reduced-size `static_plugin` build while the same build passed on
the CPU backend, the reduced-size `static_lib` build passed, and the full-op `static_plugin` build passed — the
failure required plugin + WebGPU + API-only exception catching together.

The fix is to not use exceptions for control flow in the adapters: `GetAttr()` / `GetAttrs()` now call the
non-throwing `OrtApi::KernelInfoGetAttribute*` functions directly and convert the returned `OrtStatus*` into a
`Status`. **General rule: header-only EP API adapter code must not rely on catching exceptions**, because a plugin
EP may be compiled with exception catching disabled. The remaining `catch` blocks in `include/onnxruntime/ep`
(`common.h`, `get_capability_utils.h`, `adapter/kernel_registry.h`) sit on genuinely exceptional callback
boundaries rather than on control-flow paths; when catching is disabled those exceptions still reach the outer C
API boundary and are reported, only with a less specific message.

With that fix the reduced-size `static_plugin` build passes `suite0` on the WebGPU backend in full: 2152 tests,
all passing, matching the `static_lib` control run on the identical flag set.

### Measured performance comparison

The same two binaries were compared on inference performance in Edge, using the test runner's `--perf` mode
(`test-runner-cli model <folder> -b=webgpu -e=edge -P=<n>`) against two synthetic models chosen to isolate the two
places the plugin path could plausibly cost something:

- **dispatch-bound** — 300 chained tiny elementwise ops (`Mul`/`Add`/`Sub`) on a `[1, 1024]` tensor. Almost no GPU
  work per node, so the run time is dominated by per-node host-side work, which is exactly what now goes through
  the C API adapters.
- **compute-bound** — 16 chained `[512, 512]` `MatMul`s. Host-side per-node cost is negligible against the GPU
  work.

Three runs per model per build. **The two builds must be sampled interleaved** — one round is
`plugin/dispatch, plugin/compute, lib/dispatch, lib/compute`, repeated — rather than all samples of one build
followed by all samples of the other. This machine drifts: a build that has just finished, a warm browser, or a
warm GPU shifts timings by more than the effect being measured, and sequential sampling aliases that drift onto
the build under test. The table reports the median of the three per-run P50 values.

| Metric | `static_lib` | `static_plugin` | Delta |
| --- | ---: | ---: | ---: |
| dispatch-bound, per-run P50 | 11.20 ms | 12.00 ms | +0.8 ms (+7%) |
| compute-bound, per-run P50 | 7.70 ms | 7.80 ms | +0.1 ms (within noise) |

Reading of these numbers:

- **Execution is not measurably slower once GPU work dominates.** The compute-bound samples overlap
  (`static_plugin` 7.80 / 7.80 / 7.40 against `static_lib` 7.90 / 7.00 / 7.70 — the plugin build's best sample
  beats two of the three control samples), so the difference is not resolvable at this sample count.
- **A dispatch-bound graph pays roughly 7%.** That is the cost of routing per-node kernel work through the C API
  adapters instead of a direct in-process virtual call. Unlike the compute-bound case this one does separate
  cleanly — the plugin build's slowest control sample is still faster than its own fastest sample. It only shows
  up when nodes do essentially no GPU work, which is the worst case by construction rather than a representative
  one. Amortized over ~300 nodes it is on the order of 3 µs per node — but see the caveat below before relying on
  that figure.
- **Session creation costs about 3%** (measured pre-merge: 1136 → 1172 ms and 1250 → 1290 ms for the two models).
  This covers plugin EP registration, `GetCapability` and kernel creation through the C API. The absolute figure
  is dominated by fetching and parsing the model, so the true relative cost of the plugin machinery within session
  creation is higher than 3% — but it is tens of milliseconds once, not per inference.

Two cautions on methodology:

- The *total* wall time the test runner prints for `suite0` is not a usable metric. It varied between 3 min 15 s
  and 4 min 04 s for the **same** `static_lib` binary across two runs. Only the second figure it prints — net test
  time, excluding the per-model `before all` session-creation hooks — is stable enough to compare, and by that
  measure the two builds are equal. An early single-sample comparison of the total figure suggested a 29%
  regression that does not exist.
- Timings taken with `--webgpu.profiling.mode=default` are inflated by the timestamp queries and must never be
  compared against timings taken without it.

**The per-node magnitude is not reliably established, and a controlled re-measurement is outstanding.** Three
separate measurements of the same 300-node dispatch-bound model disagree by roughly 3.7x: +0.80 ms (2.7 µs/node)
above, +1.50 ms (5.0 µs/node) in a later four-model sweep that fitted 4.76 µs/node across 150/300/600 nodes, and
+3.00 ms (10.0 µs/node) in a four-side build comparison. The `static_lib` baseline itself moved from 7.90 ms to
11.90 ms for the same model on the same machine within a day, so build-to-build and session drift dominates the
spread, and Edge's 0.1 ms `performance.now()` floor without cross-origin isolation limits resolution further. The
three runs also did not use the same statistic (per-run P50 versus median-of-round-minimum). A re-measurement
should record the commit of each arm, the full build flag set, the model definitions, the statistic, the
interleaved sampling order and the GPU submit counts, so the number can be reproduced.

What *is* robust is the sign and the mechanism. The plugin side was slower in 26 of 26 paired rounds in the sweep
and 6 of 6 in a three-way run, and GPU submit counts were byte-identical on every side, model and round, so the
cost is CPU-side per-kernel work rather than extra queue work. The same three-way run also excludes the
JavaScript side: a `mismatch` arm (plugin bundle against a `static_lib` wasm) came out level with `static_lib`, so
the `_OrtAppendExecutionProviderV2` call and the EP-name lookup are free and the cost is inside the
`static_plugin` wasm's per-kernel C++ path. **Where exactly is not yet attributed** — an `--enable_wasm_profiling`
build would give named frames, and the native root cause recorded under [Native shared-library plugin
performance](static_plugin_ep_registration_validation.md#native-shared-library-plugin-performance) is the leading
candidate.
