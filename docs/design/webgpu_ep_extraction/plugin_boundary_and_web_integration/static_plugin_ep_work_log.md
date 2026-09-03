# Static plugin EP registration — work log

Working notes for the branch `edgchen1/webgpu_static_plugin`. This is a scratch handoff document, not a
design document: the design and its rationale live in
[`static_plugin_ep_registration_design.md`](./static_plugin_ep_registration_design.md), which is the
authoritative reference for decisions D1–D7. **Delete this file before opening the PR.**

The purpose of this log is to let someone pick the work up on a different machine — in particular a machine
with a real GPU, which the original development machine did not have.

## Goal

Make the statically linked WebGPU EP build go through the **plugin EP** path rather than the internal-EP path,
via a generic "static plugin EP" registration mechanism in ORT core. The same EP sources
(`onnxruntime/core/providers/webgpu/ep/*`) then serve both a dynamically loaded plugin DLL and a build that is
statically linked into the host. The latter is what unblocks ORT Web / Emscripten, where shared library modules
are not possible.

Whether the EP is dynamically or statically **linked** is deliberately separate from whether it is a **plugin**.
Both linkage modes are supported.

## Status

The implementation is complete and committed. CI coverage was added for the new configuration, which had none.

| Area | State |
| --- | --- |
| Implementation | Done |
| Windows build + GPU tests (`--use_webgpu static_plugin`) | Green in CI |
| Shared library plugin build (`--use_webgpu shared_lib`) | Green in CI, no regression from D2/D7 |
| CUDA plugin EP (shares `include/onnxruntime/ep/*`) | Green in CI |
| Operator-level validation on a real GPU | **Done — no behavioural difference**, see [Testing](#testing) |
| Linux/GCC build (`--use_webgpu static_plugin`) | **Green in CI** (run `33030694040`) after the `array-bounds` fix — see [GCC](#gcc-and-warnings-as-errors) |
| Minimal build | Not yet exercised — see [Open items](#open-items) |
| Emscripten / ORT Web | **Builds and links**, plugin registration verified in the binary — see [ORT Web](#emscripten-and-ort-web). Not yet run in a browser |
| PR | Not opened yet |

## Branch

- Branch: `edgchen1/webgpu_static_plugin`, pushed to `git@github.com:microsoft/onnxruntime.git` (not a fork —
  this matters, because the self-hosted 1ES CI pools are not available to forks).
- Forked from `main` at `fab4691797560a22092d9bbe8cec540b5413724b`.
- The earlier commits on the branch (`dff9cc42bf` .. `800ad881ef`) are the WebGPU extraction planning documents
  and are unrelated to this workstream's code. The code work starts at `2754674229`.

| Commit | Summary |
| --- | --- |
| `2754674229` | Support registering statically linked plugin EPs, and build WebGPU EP that way |
| `32775c305a` | Add CI jobs for the WebGPU static plugin EP build |
| `49855f6ef7` | Fix GCC maybe-uninitialized errors in the EP adapter attribute accessors |
| `d47f00a0a6` | Update dynamic plugin EP test config parsing tests |
| `1980e901ec` | Record CI coverage in the static plugin EP registration design doc |
| `754e025ddc` | Add a work log for the static plugin EP branch |
| `a362107df9` | Merge `origin/main` (52 commits behind at the time; merged, not rebased) |
| `35e6f65d52` | Record real-GPU validation results in the work log |
| `a110948b1d` | Record GPU validation and the plugin-path test coverage gap in the design doc |
| `8d87c11972` | Avoid GCC 14 `-Warray-bounds` false positive in `EmitGateActivationExpr` — the batch-3 CI fix |
| `ef68884a95` | Record the complete GCC 14 warning inventory and the local reproduction recipe |

Pushed through `ef68884a95`. The push from `754e025ddc` was a fast-forward; no history has been rewritten on
this branch.

## What the change consists of

Read the design doc first. The short version, keyed to the decisions:

- **D1** — A link-time registry in ORT core (`onnxruntime/core/session/plugin_ep/ep_static_plugins.{h,cc}`),
  hand-written and `#if`-guarded, mirroring `EpLibraryInternal::CreateInternalEps`. Code generation from CMake
  and a new public C API were both considered and rejected.
- **D2** — `ORT_PLUGIN_EP_ENTRY_POINT_PREFIX` in `onnxruntime/core/providers/webgpu/ep/api.cc`. The shared
  library build must keep exporting **unprefixed** `CreateEpFactories` / `ReleaseEpFactory`, because
  `EpLibraryPlugin::Load` looks those up by name. The static build emits `WebGpu_`-prefixed symbols instead.
- **D3** — No new ABI hook for process-global teardown. `ShutdownProtobufLibrary()` is compiled out unless
  `ORT_PLUGIN_EP_OWNS_PROCESS_GLOBALS`. The legacy WebGPU cleanup block in `ort_env.cc` must stay for the
  default internal-EP build.
- **D4** — CMake option `onnxruntime_WEBGPU_STATIC_PLUGIN` plus the derived
  `onnxruntime_WEBGPU_LINKED_INTO_HOST`; surfaced as `--use_webgpu static_plugin`.
- **D5** — `ep_library_path` became optional in the dynamic plugin EP test infrastructure's
  `InitializationConfig`, so a statically registered EP can be selected without a path to load.
- **D6** — Publish-then-register in `ort_env.cc`, and `OrtEnv::m_` became a `std::recursive_mutex`. Registration
  reenters the environment (`GetSupportedDevices()` → `Api().ep.GetEnvConfigEntries()` →
  `OrtEnv::TryGetInstance()`), which self-deadlocked with a plain mutex.
- **D7** — `ORT_PLUGIN_EP_STATICALLY_LINKED` disables the manual C++ API init in the shared
  `include/onnxruntime/ep/api.h`. Do not try to force `ORT_API_MANUAL_INIT` on for everyone: MSVC's
  `#pragma detect_mismatch` requires whole-binary agreement and doing so produced roughly 50 `LNK2038` errors.

Files touched outside `docs/`:

```
.github/workflows/linux_webgpu.yml
.github/workflows/windows_webgpu.yml
cmake/CMakeLists.txt
cmake/onnxruntime.cmake
cmake/onnxruntime_providers_webgpu.cmake
cmake/onnxruntime_python.cmake
cmake/onnxruntime_unittests.cmake
include/onnxruntime/core/session/environment.h
include/onnxruntime/ep/adapter/op_kernel_info.h
include/onnxruntime/ep/api.h
onnxruntime/core/providers/webgpu/ep/api.cc
onnxruntime/core/providers/webgpu/math/{gemm,softmax,top_k}.h
onnxruntime/core/providers/webgpu/tensor/cast.h
onnxruntime/core/session/environment.cc
onnxruntime/core/session/ort_env.{h,cc}
onnxruntime/core/session/plugin_ep/ep_library_plugin.cc
onnxruntime/core/session/plugin_ep/ep_library_plugin_utils.{h,cc}      (new)
onnxruntime/core/session/plugin_ep/ep_library_static_plugin.{h,cc}     (new)
onnxruntime/core/session/plugin_ep/ep_static_plugins.{h,cc}            (new)
onnxruntime/test/framework/dynamic_plugin_ep_test.cc
onnxruntime/test/framework/inference_session_test.cc
onnxruntime/test/unittest_main/test_main.cc
onnxruntime/test/unittest_util/test_dynamic_plugin_ep.{h,cc}
tools/ci_build/{build.py,build_args.py}
```

## Building

Three WebGPU configurations exist and it is worth being able to build at least two of them, because several
bugs in this work only appeared in one of them:

```
python tools/ci_build/build.py ... --use_webgpu                # internal EP (the default, pre-existing path)
python tools/ci_build/build.py ... --use_webgpu shared_lib     # plugin EP in its own shared library
python tools/ci_build/build.py ... --use_webgpu static_plugin  # plugin EP linked into the host  <-- the new one
```

`--use_webgpu static_plugin` maps to `-Donnxruntime_USE_EP_API_ADAPTERS=ON
-Donnxruntime_WEBGPU_STATIC_PLUGIN=ON` (see `tools/ci_build/build.py`, the `--use_webgpu` handling). Both
options default to `OFF`.

`onnxruntime_USE_EP_API_ADAPTERS=ON` is **incompatible** with `onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON`
(`cmake/CMakeLists.txt` raises a `FATAL_ERROR`). The internal-EP WebGPU CI job sets the latter; the plugin jobs
must not.

### Windows gotchas

These cost real time on the original machine and are likely to recur:

- **`MAX_PATH`.** Building WebGPU under a long path such as `<repo>\build\Windows` makes Dawn's DXC build fail
  with `MSB3491`. Build into a short directory instead — `C:\b` and `C:\b2` were used for the `static_plugin`
  and `shared_lib` configurations respectively. With the Visual Studio generator the binaries land in
  `<build_dir>\<config>\<config>\`.

  Refinement from a later machine: this is really an **MSBuild** problem. With `--cmake_generator Ninja` and
  the registry's `LongPathsEnabled=1`, building at `D:\source\onnxruntime_4\build\sp` worked fine, so an
  in-repo build directory is usable if you avoid MSBuild. Note that with Ninja the binaries land in
  `<build_dir>\<config>\`, one level, not two.
- **CMake may not be on `PATH`.** On a Visual Studio install it is at
  `C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`.
- Incremental rebuild of just the EP is much faster than a full build:
  `cmake --build <build_dir> --config Debug --target onnxruntime_providers_webgpu --parallel`.

### Verifying the D2 export contract

For the **shared library** configuration, confirm the exported entry points are still unprefixed, since
`EpLibraryPlugin::Load` resolves them by name:

```
dumpbin /exports <build_dir>\<config>\<config>\onnxruntime_providers_webgpu.dll
```

Expect exactly `CreateEpFactories` and `ReleaseEpFactory`. The `static_plugin` build should instead produce
`WebGpu_`-prefixed symbols and no DLL.

## Testing

Relevant tests:

- `onnxruntime_test_all --gtest_filter=InferenceSessionTests.WebGpu*` — covers registration, device
  enumeration, and session creation end to end through the public V2 API.
- `onnxruntime_provider_test` — WebGPU EP kernel tests.

The original development machine was a **Hyper-V VM with no real GPU**. On such a host WebGPU only surfaces a
*virtual* `OrtEpDevice`, and any session that is not compile-only is rejected up front with
*"WebGPU EP was selected on a virtual GPU device…"*. So locally only these two pass:

- `InferenceSessionTests.WebGpuVirtualDeviceCompileOnlyEndToEnd`
- `InferenceSessionTests.WebGpuVirtualDeviceRejectedWithoutCompileOnly`

and `WebGpuCompileOnlySkipsFinalization` and `TestStrictShapeInference` fail. This was proven to be a property
of the host and not of static linking, by building `--use_webgpu shared_lib` on the same machine and observing
the identical pass/fail set.

**Consequence for whoever picks this up:** operator execution cannot be validated on a GPU-less machine at all.
Two ways to get real coverage:

1. **Use CI.** The new `webgpu_static_plugin_build_x64_RelWithDebInfo` job in `.github/workflows/windows_webgpu.yml`
   runs on the `onnxruntime-github-Win2022-GPU-A10` pool, which is the only GPU-equipped WebGPU pool, and it
   both builds and tests. It has passed. This is currently the primary source of operator-level truth.
2. **Run locally on a real GPU.** Nothing in the change is GPU-vendor specific; a normal Windows machine with a
   D3D12 or Vulkan capable GPU is enough.

### Results on a real GPU

This has now been done, on a machine with an NVIDIA RTX 5060 Ti. Both configurations were built from the same
tree, RelWithDebInfo, Ninja, differing **only** in the EP path:

```
python tools\ci_build\build.py --config RelWithDebInfo --build_dir <dir> --skip_submodule_sync --parallel \
  --cmake_generator Ninja --use_webgpu [static_plugin] --update --build --skip_tests \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=ON \
                        onnxruntime_ENABLE_DAWN_BACKEND_D3D12=1 onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=1
```

`onnxruntime_provider_test`, with `ONNXRUNTIME_TEST_GPU_DEVICE_ID=0`:

| | static_plugin | internal EP |
| --- | --- | --- |
| Total | 5894 | 5938 |
| Passed | 5775 | 5803 |
| Failed | 0 | 0 |
| Skipped | 119 | 135 |

**Across the 5894 tests common to both builds there were zero status differences.** This is the evidence that
routing the static build through the plugin path changes no operator behaviour.

`onnxruntime_test_all`: 1965 passed on both, no differences. All three `InferenceSessionTests.WebGpu*` pass on
the `static_plugin` build, including `WebGpuCompileOnlySkipsFinalization` — confirming the earlier failure of
that test was a property of the GPU-less host, as suspected.

Two categories of test legitimately differ between the builds; neither is a regression:

- **44 tests exist only in the internal-EP build.** 16 are `DISABLED_` and never run. The other 28 are
  white-box tests that include internal WebGPU EP headers (`WebGpuContextTest`, `ActivationCacheKeyTest`,
  `MatMul2BitsWebGpu`, `HardSwish_WebGPU`); `cmake/onnxruntime_unittests.cmake` (the
  `onnxruntime_USE_WEBGPU AND NOT onnxruntime_USE_EP_API_ADAPTERS` guard) excludes `test/providers/webgpu/*`
  from adapter builds entirely. Of these, `HardSwish_WebGPU` and `MatMul2BitsWebGpu` (12 tests) are real
  operator coverage that exists only on the internal path — a known coverage gap of the plugin path, worth
  either documenting in the design doc or closing by porting them to EP-agnostic tests.
- **The `InferenceSessionTests.WebGpu*` virtual-device tests are deliberately paired, not shared.** The
  internal build has `WebGpuEpFactoryVirtualDevice`, `WebGpuEpFactoryRejectsVirtualDeviceWithoutCompileOnly`
  and `WebGpuCompileOnlyUsesNoOpAllocator`; the plugin build has `WebGpuVirtualDeviceCompileOnlyEndToEnd` and
  `WebGpuVirtualDeviceRejectedWithoutCompileOnly`, which `inference_session_test.cc` documents in-source as the
  plugin-build counterparts. Only `WebGpuCompileOnlyUsesNoOpAllocator` has no counterpart, because it asserts
  an internal allocator *type* via `dynamic_cast`, which is inherently unobservable across the ABI boundary.

One caveat for whoever repeats this: do **not** run the two `onnxruntime_test_all` binaries concurrently.
`PathValidationTest`'s symlink tests collide over a shared temp directory and produce a spurious, and
non-deterministic, mix of pass/fail/skip. Run sequentially: all 31 pass on both builds.

There is also a `webgpu-local-testing` skill in `.agents/skills/` describing how to run WebGPU tests on Linux
without a GPU using Mesa lavapipe (software Vulkan). Note its stated limitation: any graph containing `MatMul`
crashes lavapipe, so it only validates host-side enforcement and shape logic.

### Not covered by any test yet

Re-registration — register, run, unregister, re-register — is untested and is not reachable from the existing
test binaries, because `OrtEnv` is a refcounted process singleton that the unit test main pins for the lifetime
of the process.

## CI

No pre-existing leg built the `static_plugin` configuration, since both CMake options default to `OFF`. Every
WebGPU job was either internal EP (`--use_webgpu`) or shared library plugin (`--use_webgpu shared_lib`). Two
jobs were added:

- `webgpu_static_plugin_build_x64_RelWithDebInfo` in `.github/workflows/windows_webgpu.yml` — builds **and
  tests** on the A10 GPU pool. It deliberately omits `onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON` (incompatible,
  see above) and deliberately drops the `--disable_rtti` / `--enable_lto` flags that the shared library job
  carries, since those describe how the plugin ships as a DLL and here they would apply to all of ORT.
- `build-linux-webgpu-static-plugin-x64-release` in `.github/workflows/linux_webgpu.yml` — build only, mirroring
  its sibling job. This is the only **GCC** coverage of the EP adapters, and the only coverage of the
  `cmake/onnxruntime_python.cmake` change, which is why `--build_wheel` is kept on it.

`linux_webgpu.yml` is build-only by design; its test job is commented out with a pre-existing "currently
failing" TODO.

### Running CI on this branch

The workflows only trigger automatically on `push`/`pull_request` for `main`, `rel-*`, and
`plugin-ep-webgpu/rel-*`, so **pushing this branch triggers nothing**. Dispatch manually:

```
gh workflow run windows_webgpu.yml     --repo microsoft/onnxruntime --ref edgchen1/webgpu_static_plugin
gh workflow run linux_webgpu.yml       --repo microsoft/onnxruntime --ref edgchen1/webgpu_static_plugin
gh workflow run windows_cuda_plugin.yml --repo microsoft/onnxruntime --ref edgchen1/webgpu_static_plugin
```

`gh workflow run --ref <branch>` reads the workflow file **from that ref**, so jobs added on the branch do run
even though they are not on `main` yet. This was confirmed empirically.

`windows_cuda_plugin.yml` matters because the CUDA plugin EP shares `include/onnxruntime/ep/*` with WebGPU, so
it is the regression check for the D5 and D7 changes.

### Retrieving logs

`gh run view --log-failed` and `gh run view --job <id> --log` both returned **zero bytes** for these runs. Use
the REST API instead:

```
gh api repos/microsoft/onnxruntime/actions/runs/<run_id>/jobs --jq '.jobs[] | select(.conclusion=="failure") | "\(.id) \(.name)"'
gh api repos/microsoft/onnxruntime/actions/jobs/<job_id>/logs > log.txt
```

When grepping the result, search for `FAILED:` **case-sensitively**. A case-insensitive search for "failed" or
a search for `error:` drowns in benign CMake and Dawn feature-probe output such as
`-- Performing Test ... - Failed`. The same trap applies to local build logs: `grep -c "error:"` matches Dawn's
own configure line `-- DAWN Werror: OFF`, which looks alarming in a summary count and is nothing.

### Run history

| Run | Workflow | Result |
| --- | --- | --- |
| 32914301463 | ONNX Runtime WebGPU Builds | Success, all 6 jobs including the new one |
| 32914304408 | Linux WebGPU CI | Failure — GCC `maybe-uninitialized`, batch 1 |
| 32914307098 | CUDA Plugin Windows CI | Failure — `ParseInitializationConfigRejectsMissingRequiredFields` |
| 32996643911 | Linux WebGPU CI | Failure — GCC `maybe-uninitialized`, batch 2 |
| 32996646394 | CUDA Plugin Windows CI | Success |
| 33009901327 | Linux WebGPU CI | Failure — GCC `array-bounds`, batch 3. Internal-EP job on the same run passed. |
| 33030694040 | Linux WebGPU CI | **Success** — both jobs. Confirms the `array-bounds` fix; first green GCC run on this branch. |
| 33030695742 | ONNX Runtime WebGPU Builds | Dispatched on `ef68884a95` — re-run after the merge from `main` |
| 33030697208 | CUDA Plugin Windows CI | Dispatched on `ef68884a95` — re-run after the merge from `main` |

## GCC and warnings as errors

This was the only substantial problem the new CI surfaced, and it will bite again, so it is worth understanding
rather than pattern-matching.

The EP adapter types are defined **entirely inline in headers**. The in-tree `OpKernelInfo` hides its
accessors behind out-of-line definitions in a `.cc`, so GCC cannot see their failure and exception paths. With
the adapters, it can — and it then warns about values that are only ever read once the call has succeeded. The
warning is a false positive, but the build uses `-Werror`.

This had never been observed before simply because the EP-adapters path had never been compiled with GCC: the
existing Linux WebGPU CI builds the internal EP, and the shared library plugin builds are Windows/MSVC-only.
The CUDA plugin EP hit the identical problem earlier and solved it with a target-scoped suppression in
`cmake/onnxruntime_providers_cuda_plugin.cmake`.

Two batches of failures appeared, because ninja stops early and only reports the targets that were already in
flight — **do not assume the first batch is the complete list**. A third batch, of a different warning class,
appeared later still. The lesson is the general one: **each CI round trip reveals only one batch**, so prefer a
local GCC build (or a throwaway dispatch with `--compile_no_warning_as_error`) to enumerate all of them at once.

- Batch 1, fixed at the source in `49855f6ef7`: uninitialized locals that are passed by pointer and read back,
  and the temporaries inside `GetAttrOrDefault` / `GetAttr` in `include/onnxruntime/ep/adapter/op_kernel_info.h`.
  Value-initializing them (`T tmp{}`) is correct and worth keeping on its own merits.
- Batch 2: false positives deep inside `absl::InlinedVector`, for `TensorShapeVector` locals in
  `tensor/pad.cc` and `tensor/unsqueeze.h` that are always properly constructed. These cannot be fixed at the
  source. Handled by applying the same `-Wno-maybe-uninitialized` suppression the CUDA plugin EP uses, scoped
  to the `onnxruntime_providers_webgpu` target, GCC only, and only when `onnxruntime_USE_EP_API_ADAPTERS` is on
  so that non-adapter builds stay strict.
- Batch 3, in run `33009901327`, **fixed at the source**: `-Werror=array-bounds`, a *different* warning class, at
  `onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_mlp.cc:47` in `EmitGateActivationExpr`:

  ```
  ‘void* __builtin_memcpy(...)’ forming offset [32, 48] is out of the bounds [0, 32]
  of object ‘<anonymous>’ with type ‘std::__cxx11::basic_string<char>’
  ```

  GCC 14 at `-O3`, inlining the `std::string{gate_var} + " * (one / (one + exp(-" + std::string{gate_var} + ")))"`
  temporary-move concat chain through `std::operator+(basic_string&&, const char*)`. A false positive. Note that
  the batch-2 `-Wno-maybe-uninitialized` suppression **is** working — it is present on the failing command line
  and the batch-2 files compiled clean. The file is pre-existing (added by PR #28280, `88ca23fcd5`), not authored
  by this branch; the adapters PCH pulls in `include/onnxruntime/ep/adapters.h`, which changes inlining and is
  why only the adapters config trips it. Fixed by building the expression with ORT's `MakeString` instead of a
  `std::string` concat chain — a single-site source fix, preferred over extending the target-scoped suppression
  with `-Wno-array-bounds`, which would have blinded the whole provider to a genuinely useful warning.

### The complete GCC 14 warning inventory

The batch-at-a-time discovery loop was finally closed by enumerating **every** warning in one pass: a full
container build with `--compile_no_warning_as_error`, then grouping the log by warning class. The result:

| Warning class | Count | Where |
| --- | --- | --- |
| `-Wattributes` | 276 | Dawn generated code (`dawn_proc.cpp`) |
| `-Wunused-parameter` | 55 | ONNX (`onnx/defs/**`) |
| `-Wdangling-pointer=` | 3 | Dawn / Tint |
| `-Wchanges-meaning` | 3 | Dawn (`SystemEvent.h`) |
| `-Wstringop-overflow=` | 2 | flatbuffers (`reflection.cpp`) |
| `-Warray-bounds=` | **1** | **ORT — `matmul_nbits_mlp.cc:47`** |
| `-Wstringop-overread` | 1 | telemetry (`sqlite3_retail.c`) |

**Exactly one warning in the whole build is attributable to ORT's own source**, and it is the batch-3 failure.
Everything else is third-party, in targets that do not compile with `-Werror`. So there is no batch 4: fixing
this one site is expected to turn the leg green.

### Reproducing the CI compiler locally

Worth the setup cost — it turns a ~28 minute CI round trip per batch into a local loop, and it is the only way
to enumerate all batches at once.

The CI base image (`onnxruntimebuildcache.azurecr.io/.../cpu_x64_almalinux8_gcc14`) requires ACR auth and is
**not pullable**. An equivalent image is easy to build, and reproduces the compiler exactly
(GCC 14.2.1 20250110, Red Hat) — `almalinux:8` + `epel-release` + `gcc-toolset-14` + `python3.11`, with
`cmake ninja packaging numpy setuptools wheel` from pip. Notes:

- **Python 3.11 is required, not the distro's 3.9** — `tools/ci_build/build_args.py` uses a `match` statement
  (3.10+). CI itself uses `/opt/python/cp310-cp310/bin`.
- `--allow_running_as_root` is required inside the container.
- **No X11 dev packages are needed**, because `cmake/external/onnxruntime_external_deps.cmake` sets
  `DAWN_USE_X11 OFF` unconditionally. This is what keeps the image minimal.
- Clone into the WSL native filesystem, not `/mnt/d` — building over the 9p mount is crippling. Only
  `cmake/external/onnx` and `cmake/external/libprotobuf-mutator` need initialising for a native Linux build.

To re-check a single file without paying for a whole build, override `CXX_FLAGS` on the generated makefile
rule, which lets you promote warnings back to errors against an existing build tree:

```bash
FLAGS=$(sed -n 's/^CXX_FLAGS = //p' CMakeFiles/onnxruntime_providers_webgpu.dir/flags.make)
make -f CMakeFiles/onnxruntime_providers_webgpu.dir/build.make \
     CMakeFiles/onnxruntime_providers_webgpu.dir/src/<path>/matmul_nbits_mlp.cc.o \
     CXX_FLAGS="$FLAGS -Werror"
```

Always run the **negative control** — rebuild the unfixed source through the same harness and confirm it still
errors. Otherwise a harness that silently fails to exercise the warning looks exactly like a successful fix.

### Verification of the fix

Both halves were checked, because each one alone is inconclusive:

1. **Single-object recheck.** With the fix, `matmul_nbits_mlp.cc.o` compiles clean under `-Werror`. Reverting the
   file and rebuilding through the identical harness reproduces the exact CI error, which is what proves the
   harness was actually exercising the warning.
2. **Full build with `-Werror` on**, i.e. the real CI configuration with `--compile_no_warning_as_error` dropped:
   completes to 100%, links `onnxruntime_provider_test`, zero compiler errors. Before trusting this, confirm
   `-Werror` really is on the ORT targets — `sed -n 's/^CXX_FLAGS = //p' CMakeFiles/<target>.dir/flags.make`
   should show it for `onnxruntime_providers_webgpu`, `onnxruntime_providers` and `onnxruntime_session` — and
   confirm the fixed file appears in the build log, so it was recompiled rather than served from a stale object.

The 334 warnings that remain in that build are all third-party, in targets that do not use `-Werror`.

## Emscripten and ORT Web

This is the motivating use case for the whole change, so it was built locally before opening the PR. **It builds
and links, and the plugin registration is verifiably present in the output.**

Note what ORT Web's own CI does *not* cover today: the WebGPU WASM jobs in
`.github/workflows/linux-wasm-ci-build-and-test-workflow.yml` pass a bare `--use_webgpu`, and bare `--use_webgpu`
defaults to `static_lib` (`tools/ci_build/build_args.py`, `const="static_lib"`). So every existing WASM job still
builds the **internal** EP. Nothing in CI exercises WASM + `static_plugin`; this branch only makes the
configuration available, it does not switch ORT Web over to it.

The CMake side was already written with Emscripten in mind — `cmake/onnxruntime_providers_webgpu.cmake` rejects
`shared_lib` there with `"WebGPU EP shared library build is not supported on Emscripten. Please use
'--use_webgpu static_plugin'."`

### Building it

Same container as the GCC 14 reproduction (see [above](#reproducing-the-ci-compiler-locally)), plus:

- `git submodule update --init --recursive cmake/external/emsdk` — not needed for a native Linux build, required
  here. `build.py` installs and activates the toolchain itself.
- **`node` must be on `PATH`.** `cmake/external/onnxruntime_webassembly.cmake` includes `node_helper.cmake`,
  which does a hard `find_program(NODE_EXECUTABLE ...)` and fails configure with
  `Could not find NODE_EXECUTABLE using the following names: node.exe, node`. CI gets node from
  `actions/setup-node`; the AlmaLinux image has none. emsdk bundles one — point `PATH` at
  `cmake/external/emsdk/node/<version>/bin`.

```
docker run --rm -v /home/edch/ort:/src -v /home/edch/ortwasm:/build -w /src ort-gcc14:local \
  bash -lc "export PATH=/src/cmake/external/emsdk/node/22.16.0_64bit/bin:\$PATH && \
  python3.11 tools/ci_build/build.py --build_dir /build --config Release --skip_submodule_sync --parallel \
  --allow_running_as_root --build_wasm --enable_wasm_simd --enable_wasm_threads \
  --enable_wasm_api_exception_catching --use_webgpu static_plugin --use_webnn \
  --target onnxruntime_webassembly --skip_tests"
```

Result: configure clean, zero compile failures, links `ort-wasm-simd-threaded.asyncify.mjs` +
`.wasm` (~32 MB). Roughly 35 minutes including the emsdk toolchain download and sysroot generation.

### Verifying the plugin path actually linked

A successful build alone proves little here, because the registration is behind
`#if defined(USE_WEBGPU) && defined(ORT_WEBGPU_STATIC_PLUGIN)` in
`onnxruntime/core/session/plugin_ep/ep_static_plugins.cc`. If that define failed to reach ORT core, the file
would still compile — `CreateStaticPluginEpLibraries()` would just return an empty vector, and the build would
be just as green while registering nothing.

The decisive check is that ORT core holds an **undefined** reference to the prefixed entry points, using the
emsdk LLVM tools:

```
NM=cmake/external/emsdk/upstream/bin/llvm-nm
$NM <build>/Release/libonnxruntime_session.a          | grep WebGpu_   # expect: U WebGpu_CreateEpFactories, U WebGpu_ReleaseEpFactory
$NM <build>/Release/libonnxruntime_providers_webgpu.a | grep WebGpu_   # expect: T WebGpu_CreateEpFactories, T WebGpu_ReleaseEpFactory
```

Observed exactly that: `U` in core, `T` in the provider, and the link resolved them. That closes the loop —
the `#if` was live, the entry point prefixing worked, and `EpLibraryStaticPlugin` is in the binary.

### Running it in a browser

The wasm binary has now been **run**, on a real GPU, in Edge. A dedicated harness lives in `build/bench_web/`
(untracked scratch, not part of the repo):

- `make_side.ps1 -Side <name>` builds one **self-consistent "side"**: it copies that side's
  `ort-wasm-simd-threaded.asyncify.{wasm,mjs}` into `js/web/dist`, checks out the matching
  `js/web/lib/wasm/session-options.ts`, runs `npm run build -- --bundle-mode=perf --webgpu-ep`, and snapshots the
  result with a `MANIFEST.txt` recording git SHA, source provenance and SHA-256 of every artifact.
- `serve.js` serves the snapshots on port 8099; `bench.js` is the page.
- `run_bench.ps1 -Sides @(...)` drives an interleaved (ABBA / cyclic-rotation) run across N sides.

Two properties of the harness matter more than the timings:

- **`--webgpu-ep` is mandatory and fails silently if omitted.** Without it `js/web/script/build.ts` sets
  `DISABLE_WEBGPU=true` and the session quietly runs on **JSEP** instead, producing a plausible-looking but
  completely irrelevant number.
- **A bundle and a wasm are a matched pair.** `build.ts` has an esbuild `onLoad` hook that *inlines the `.mjs`
  loader into the bundle*, so a bundle from one build silently carries another build's loader. Sides must never
  be mixed by hand.

### Verifying the plugin path is actually taken at runtime

Linking is not execution, and the failure mode here is silent, so the runtime path was pinned down with
controls rather than by reading code:

| control | bundle | wasm | result |
| --- | --- | --- | --- |
| `plugin` | plugin | `static_plugin` | runs, correct output |
| `mismatch` | plugin | `static_lib` | **runs, correct output** — the silent trap |
| `mismatch2` | lib | `static_plugin` | fails: "WebGPU execution provider is not supported in this build" |
| bogus EP name | patched | `static_plugin` | fails: "No execution provider device is registered for `…BOGUS`" |

The bogus-name failure goes on to list the registered devices — `CPUExecutionProvider` and
`WebGpuExecutionProvider`. That control is the load-bearing one: it proves the V2 name lookup is real and
enforced, so a *successful* `plugin` run necessarily resolved a genuinely registered plugin EP device.

The page also patches `GPUQueue.prototype.submit` to count GPU submissions. That counter has been **perfectly
deterministic** across every run, which makes it a far better endpoint than any timing statistic.

### Measured cost of the plugin path on the web build

Comparing `static_lib` against `static_plugin` wasm built from the *same* commit — the two `build.py` argument
dumps are identical except `use_webgpu='static_lib'` vs `'static_plugin'` — over 6-8 interleaved rounds per
model, taking the median of each round's minimum:

| model | nodes | `static_lib` | `static_plugin` | delta | predicted @ 4.76 us/node | residual |
| --- | --- | --- | --- | --- | --- | --- |
| `bench_compute` | 16 | 5.00 ms | 5.00 ms | 0.00 ms | 0.08 ms | -0.08 ms |
| `bench_dispatch_150` | 150 | 4.30 ms | 5.10 ms | +0.80 ms | 0.71 ms | +0.09 ms |
| `bench_dispatch` | 300 | 7.90 ms | 9.40 ms | +1.50 ms | 1.43 ms | +0.07 ms |
| `bench_dispatch_600` | 600 | 13.80 ms | 16.60 ms | +2.80 ms | 2.86 ms | -0.06 ms |

The plugin side was slower in **26 of 26** paired rounds. A single through-origin constant of **~4.8 us of CPU
per kernel node** fits all four points to within one 0.1 ms timer tick, and **GPU submit counts were
byte-identical on every side, model and round** (3 per run for compute; 11/20/39 for 150/300/600). So this is
pure per-kernel CPU dispatch overhead, not extra queue work. The 16-node null is not evidence of no cost — the
predicted 0.08 ms simply sits below the timer floor.

A three-way run isolated where the cost is **not**: the `mismatch` side (plugin bundle, `static_lib` wasm) came
out level with `lib` (median delta -0.20 ms, a 3/6 sign split), while `plugin` was +1.60 ms in 6/6 rounds.
The JS-side `_OrtAppendExecutionProviderV2` call and EP-name lookup are therefore free; the cost is inside the
`static_plugin` wasm's per-kernel C++ path. Where exactly is still unknown — 4.8 us is roughly 10k cycles,
far too much for an indirect call, so it points at real per-kernel work. An `--enable_wasm_profiling` build
would give named frames.

Timer resolution is 0.1 ms: Edge coarsens `performance.now()` without cross-origin isolation. Enabling
COOP/COEP would give ~5 us resolution and was deliberately *not* turned on mid-experiment, to keep all numbers
comparable.

### Correction: what the shared-allocator commit does and does not explain

An earlier reading of these results as "refuting" the design doc's +36% figure was **wrong**, and the mistake is
worth recording. Commit `b876d290cd` ("Defer zero-initialize submission in the WebGPU plugin EP shared
allocator") is an **ancestor** of the benchmarked branch head, so every `plugin` number above is already
*plugin-with-the-fix*. Parity on the compute-bound model with identical submit counts is therefore
**consistent with that fix working**, not evidence against the original measurement.

A second retraction: the claim that the earlier *native* A/B was "null by construction" because
`cmake/onnxruntime_providers_webgpu.cmake` excludes the `ep/` folder from non-plugin builds was also wrong. The
build logs show both native sides were `use_webgpu='shared_lib'`, which **does** compile `ep/factory.cc`. That
null was genuine.

What remains genuinely untested is whether the deferral matters on **web/wasm** at all. That was settled by
a four-side experiment — `main_lib` (merge-base `2fee06a0c1`), `lib`, `plugin_nofix` (branch head with
`b876d290cd` reverted, commit `a6be215b31`) and `plugin` — each built from its **own worktree and its own build
directory**, with no reuse and no `--skip_submodule_sync`. The **primary endpoint is the deterministic submit
count, not timing**: the commit's own mechanism claim is "a queue submit for every intermediate buffer allocated
mid-run", so `plugin_nofix` must show more submits per run than `plugin` if it does anything at all.
`bench_dispatch` has 299 intermediates against `bench_compute`'s 15, making it ~20x the detector.

### Result: `b876d290cd` has no measurable effect on the web build

The submit counter is perfectly deterministic — the same value in all 8 rounds of every side, both models:

| model | dispatches/run | submits/run: `main_lib` | `lib` | `plugin_nofix` | `plugin` |
| --- | --- | --- | --- | --- | --- |
| `bench_compute` | 16 | 3 | 3 | 3 | 3 |
| `bench_dispatch` | 300 | 20 | 20 | 20 | 20 |

**Reverting the commit changes nothing.** On `bench_dispatch`, 300 dispatches with 299 intermediates still batch
into exactly 20 submits whether the deferral is present or not. The mechanism the commit describes — one submit
per intermediate buffer — simply **does not occur on the wasm build**, so there is nothing there for it to defer.

Timing agrees, and is noise: `plugin - plugin_nofix` med `dmin` is **+1.70 ms (6/8 rounds *slower*)** on
`bench_dispatch` but **-0.30 ms (3/8)** on `bench_compute`. The sign flips between models, which is the
signature of build-to-build variation (code layout / icache) rather than a real effect. Note the direction: the
side *with* the fix is if anything marginally slower on the dispatch-heavy model.

Combined with the earlier genuine native A/B null, `b876d290cd` has **no demonstrated benefit on any platform
tested**, and no correctness change was identified. It should not be landed on its own merits.

Two other contrasts fall out of the same run, on the `medmin` delta:

| pair | `bench_compute` | `bench_dispatch` | reading |
| --- | --- | --- | --- |
| `lib - main_lib` | 0.00 ms (0.0%) | -0.10 ms (-0.8%) | the branch does **not** regress the static_lib path |
| `plugin - lib` | +0.30 ms (+5.8%) | +3.00 ms (+25.2%) | cost of the plugin path itself |
| `plugin - main_lib` | +0.30 ms (+5.8%) | +2.90 ms (+24.2%) | **total ship delta vs. `main`** |

The `lib` vs `main_lib` null is the important control: it isolates the plugin-path cost from "anything else the
branch changed". The isolation is unusually clean here because the two sides' `ort.all.min.js` came out
**byte-identical** (SHA-256 `79391F71...`), the merge-base and `main` blobs of `session-options.ts` being the
same — so the *only* difference between those two sides is the `.wasm`. The same holds for
`plugin_nofix` vs `plugin` (bundle `AB2F004F...`). All four `.wasm` hashes were confirmed distinct.

## Open items

1. ~~**Fix the Linux leg.**~~ **Done.** The batch-3 `array-bounds` break is fixed at the source, and a local
   GCC 14 build confirmed it was the *only* ORT-owned warning in the whole build (see
   [The complete GCC 14 warning inventory](#the-complete-gcc-14-warning-inventory)). CI run `33030694040`
   is green on both jobs.
2. ~~**Operator-level validation on a real GPU.**~~ **Done** — see [Results on a real GPU](#results-on-a-real-gpu).
   No behavioural difference. Remaining sub-item: decide whether to close the 12-test
   `HardSwish_WebGPU` / `MatMul2BitsWebGpu` coverage gap on the plugin path, or document it as known.
3. ~~**Minimal build.**~~ **Resolved.** Exactly one pipeline combines an extended-minimal build with WebGPU —
   `webgpu_minimal_build_edge_build_x64_RelWithDebInfo` in `.github/workflows/windows_webgpu.yml`, which passes a
   bare `--use_webgpu` (internal EP) and is green, so the `#if !defined(ORT_MINIMAL_BUILD)` guard in `ort_env.cc`
   is already exercised in CI. No ORT Web or wasm job uses a minimal build at all. The remaining hole — minimal
   plus `static_plugin` linking the EP in without registering it — is now a configure-time `FATAL_ERROR` in
   `cmake/CMakeLists.txt`, verified to fire on that combination and not on a non-minimal `static_plugin`
   configure. Lifting the restriction properly is follow-up work, recorded in D4 of the design doc.
4. ~~**Emscripten / ORT Web.**~~ **Builds, links, and now runs in a browser** on a real GPU — see
   [Running it in a browser](#running-it-in-a-browser). Registration is verified in the binary *and* at runtime.
   Open sub-item: the plugin path costs **~4.8 us of CPU per kernel node** on the web build
   ([measurement](#measured-cost-of-the-plugin-path-on-the-web-build)); the source of that cost is not yet
   identified and needs an `--enable_wasm_profiling` build to attribute.
5. **Follow-up cleanup** listed at the end of the design doc, including removing the WebGPU special cases that
   only exist while it is still an "internal" EP.
6. **Open the PR.**

## Environment notes

Specific to the original machine, but the class of problem generalises.

- The working copy `C:\source\ort-webgpu-static-plugin` is a **linked git worktree** of `C:\source\onnxruntime`.
  `.git` is therefore a *file*, not a directory. Do not write scratch files into it.
- Linting is `lintrunner`, driven by the `ort-lint` skill. `lintrunner -a` can report "ok No lint issues" and
  "Successfully applied all patches" in the same run, so always re-check `git status` afterwards.
- The repository's own pre-commit hook lives in `.githooks/pre-commit` (this is not the `pre-commit` framework)
  and is enabled with `git config core.hooksPath .githooks`. It **silently does nothing if `lintrunner` is not
  on `PATH`**, so activate the virtual environment before committing or the commit goes through unlinted.
