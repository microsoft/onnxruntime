# Static Plugin EP Registration: Validation

Evidence for [Static Plugin EP Registration](static_plugin_ep_registration_design.md). That document defines the
design and its contracts; this one records what was built, what was run, and what is still untested.

## Test results

Verified on Windows with `--use_webgpu static_plugin`, Debug:

- `onnxruntime_test_all` and `onnxruntime_provider_test` build and link.
- The statically linked WebGPU plugin EP is registered by ORT core during `Ort::Env` creation, and its `OrtEpDevice`
  is enumerated and selectable, confirming the D6 publish-then-register ordering resolves the self-deadlock.
- `InferenceSessionTests.WebGpuVirtualDeviceCompileOnlyEndToEnd` and
  `InferenceSessionTests.WebGpuVirtualDeviceRejectedWithoutCompileOnly` pass, which covers session creation and EP
  instantiation end to end through the public V2 API on the static plugin path.

Operator execution was initially not verifiable locally: the original development machine was a virtual machine with
no real GPU, so only the virtual WebGPU device was surfaced and non-compile-only sessions were rejected up front. This
was confirmed to be a property of the host rather than of static linking by building the shared library plugin
configuration (`--use_webgpu shared_lib`) on the same host and running the same tests: it passed and failed exactly
the same set (`WebGpuVirtualDeviceCompileOnlyEndToEnd` and `WebGpuVirtualDeviceRejectedWithoutCompileOnly` pass;
`WebGpuCompileOnlySkipsFinalization` and `TestStrictShapeInference` fail with the identical virtual-device rejection
message).

This gap has since been closed on a machine with a real GPU (NVIDIA RTX 5060 Ti). The `static_plugin` and internal-EP
configurations were built from the same tree, RelWithDebInfo, differing only in the EP path, and their test results
compared:

- `onnxruntime_provider_test`: 0 failures in both. **Across the 5894 tests common to both builds there were zero
  status differences.**
- `onnxruntime_test_all`: 1965 passed in both, no differences. All three `InferenceSessionTests.WebGpu*` pass on the
  static plugin build, including `WebGpuCompileOnlySkipsFinalization`, confirming its earlier failure was indeed a
  property of the GPU-less host.

## Known test coverage difference on the plugin path

`cmake/onnxruntime_unittests.cmake` gates the `test/providers/webgpu/*` sources on
`onnxruntime_USE_WEBGPU AND NOT onnxruntime_USE_EP_API_ADAPTERS`, so those tests are **absent** from any adapter
build, including `static_plugin`. They are white-box tests that include internal WebGPU EP headers and therefore
cannot compile against the adapter boundary by construction.

Of the 44 tests this excludes, 16 are `DISABLED_` and never run, and most of the rest assert on EP internals
(`WebGpuContextTest`, `ActivationCacheKeyTest`). However, 12 of them — `HardSwish_WebGPU` and `MatMul2BitsWebGpu` —
are genuine operator tests, and they currently run **only** on the internal EP path. This is a known coverage gap of
the plugin path. Closing it would mean rewriting them as EP-agnostic operator tests so both paths execute them; that
is deliberately left out of this change to keep it focused.

Separately, the `InferenceSessionTests.WebGpu*` virtual-device tests are deliberately *paired* rather than shared
between the two builds, since each drives a different factory. `WebGpuVirtualDeviceCompileOnlyEndToEnd` and
`WebGpuVirtualDeviceRejectedWithoutCompileOnly` are the plugin-build counterparts of `WebGpuEpFactoryVirtualDevice`
and `WebGpuEpFactoryRejectsVirtualDeviceWithoutCompileOnly`. Only `WebGpuCompileOnlyUsesNoOpAllocator` has no
counterpart, because it asserts an internal allocator *type* through `dynamic_cast`, which is inherently
unobservable across the ABI boundary.

The shared library plugin configuration was also re-verified after the D7 change, since `include/onnxruntime/ep/api.h`
is shared with the CUDA plugin EP. It builds, and `onnxruntime_providers_webgpu.dll` still exports exactly the
unprefixed `CreateEpFactories` and `ReleaseEpFactory` required by `EpLibraryPlugin::Load`, per D2.

## CI coverage

No pre-existing CI leg built this configuration, since both new CMake options default to `OFF`. Three legs were
added:

- `webgpu_static_plugin_build_x64_RelWithDebInfo` in `.github/workflows/windows_webgpu.yml`, which builds and runs the
  tests on a GPU-equipped pool. This closes the operator execution gap above. Note that it must not set
  `onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON`, which is incompatible with `onnxruntime_USE_EP_API_ADAPTERS`.
- `build-linux-webgpu-static-plugin-x64-release` in `.github/workflows/linux_webgpu.yml`, build only, mirroring its
  sibling job. This is the only GCC coverage of the EP adapters, and the only coverage of the `onnxruntime_python.cmake`
  change, hence `--build_wheel` is kept.
- A build-only `--use_webgpu static_plugin` step in the `wasm_Release` job of `.github/workflows/web.yml`, which
  keeps the Emscripten registration path compiling. It publishes no artifacts; see
  [ORT Web Static Plugin Migration](ort_web_static_plugin_migration.md#emscripten-and-ort-web) for why, and for what
  that leg does and does not prove.

The first run of the Linux job surfaced `-Werror=maybe-uninitialized` errors. The EP adapter `OpKernelInfo` defines its
attribute accessors inline in the header, so GCC can see their failure and exception paths, whereas the in-tree
`OpKernelInfo` hides them behind an out-of-line definition. Any value that is only read when the attribute lookup
succeeded must be value-initialized.

More generally, this job is the first time the EP adapters have ever been compiled with GCC, and it has surfaced
several successive batches of `-Werror` diagnostics — including false positives inside `absl::InlinedVector` and, in
optimized builds, `-Werror=array-bounds` on `std::string` concatenation chains that the adapter headers cause to be
inlined more aggressively. Because ninja stops at the first failure and reports only the targets already in flight,
**each CI run reveals just one batch**. When touching this configuration, enumerate the full set locally in a single
pass — an AlmaLinux 8 container with `gcc-toolset-14` reproduces the CI compiler exactly — rather than discovering
them one CI round trip at a time. Fixes that are correct on their own merits are applied at the source; irreducible
false positives use a suppression scoped to the `onnxruntime_providers_webgpu` target, GCC only, and conditional on
`onnxruntime_USE_EP_API_ADAPTERS`, so non-adapter builds stay strict. This mirrors what the CUDA plugin EP does in
`cmake/onnxruntime_providers_cuda_plugin.cmake`.

That full local enumeration has since been done, and it is worth recording what it found: of the ~340 warnings GCC 14
emits across the whole `static_plugin` build, **exactly one is attributable to ORT's own source** — the
`-Warray-bounds` false positive in `contrib_ops/webgpu/quantization/matmul_nbits_mlp.cc`, fixed by building the
expression with `MakeString` instead of a `std::string` concat chain. Everything else lives in Dawn, ONNX,
flatbuffers, or the telemetry dependency, in targets that do not use `-Werror`. The practical lesson is that the
adapters do not make ORT's code broadly warning-prone; they perturb inlining at a small number of specific sites,
so a targeted source fix is almost always available and preferable to a blanket target-wide suppression.

When validating such a fix locally, always rebuild the *unfixed* source through the same harness first and confirm it
still errors. A harness that silently fails to exercise the warning is indistinguishable from a successful fix.


## Native shared-library plugin performance

The latency criterion covers two shapes of plugin, and the ORT Web numbers under [Measured performance
comparison](ort_web_static_plugin_migration.md#measured-performance-comparison) settle only the static one. A
native `--use_webgpu shared_lib` A/B against a built-in baseline measured the plugin slower by ~8% on a
dispatch-bound model and 4-6% on Qwen3.5-0.8B, fitting roughly **67 µs per `Run` plus 1.2 µs per node**. The
reproducibility caveat recorded for the web numbers applies here too; these figures should be re-taken with the
commit, flags and harness recorded.

One control is what licenses reading the delta as boundary cost at all: matching build flags do not prove two
executables perform alike, so both binaries were also run on the CPU EP, where no plugin is loaded. That gave a
ratio of minima of 1.001 — but only single-threaded. Run with default threading the same control showed the
plugin executable 24% *faster*, which would have been actively misleading.

**Root cause: redundant per-kernel work, not boundary crossing.** 1.2 µs/node is ~4,400 cycles, far more than a
cross-DLL indirect call can account for, so both arms were CPU-profiled. `ep::adapter::CreateTensorFromApiValue`
rebuilds an `onnxruntime::Tensor` — nine C API round trips, an allocator-name `std::string`, a `TensorShape`
allocation — for every input and output of every node on every `Run`, because the adapter's `OpKernelContext` is
constructed fresh per `Compute()` and caches nothing beyond it. The adapter symbols are absent from the built-in
profile entirely, `Tensor`/`TensorShape` work doubles, and the extra CPU (+0.53 ms/rep) accounts for the entire
extra wall latency (+0.42 ms/rep).

That makes the cost look **reducible rather than intrinsic**, which matters for how the latency criterion is
settled: negotiating a tolerance around a fixable defect would be the wrong order. The obvious candidates are
caching the `Tensor` wrapper across `Compute()` calls, and avoiding the allocator-name `std::string` and the
`TensorShape` allocation on the per-node path. This is also the leading candidate for the unattributed web
per-node cost, since both boundaries charge per kernel node.

Two gaps remain in the native numbers. Session-creation cost was not compared — the plugin arm must also discover
and load the DLL, so it is likely worse in relative terms. And both arms ran with default WebGPU options; graph
capture in particular targets exactly the per-dispatch cost identified above and is worth a follow-up.

**The plugin path silently loses the WebGPU EP's `Api` profiling events.** The built-in arm emits 15,300 `Api`
category events under `-p` and the plugin arm emits none. This is an observability regression independent of
latency, and it also means ORT's own profiler cannot perform this A/B: under `-p` the built-in arm measures
*slower* and the result inverts.
