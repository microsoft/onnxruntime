# Static Plugin EP Registration

Status: Detailed design supporting
[Plugin Boundary and Web/Wasm Integration](plugin_boundary_and_web_integration_workstream.md).

## Purpose

Define how a plugin execution provider (EP) that is linked into the host binary is registered with ONNX Runtime, so
that statically linked and dynamically loaded plugin EPs share one provider implementation and one public API
boundary.

WebGPU is the first consumer. The immediate deliverable is to move the statically linked WebGPU build off
`EpLibraryInternal` and onto the plugin EP path.

## Background

ORT has two `EpLibrary` implementations today:

- `EpLibraryPlugin` (`onnxruntime/core/session/plugin_ep/ep_library_plugin.cc`) loads a shared library, resolves the
  `CreateEpFactories` and `ReleaseEpFactory` symbols, and drives the factory lifecycle.
- `EpLibraryInternal` (`onnxruntime/core/session/plugin_ep/ep_library_internal.cc`) wraps an in-tree
  `IExecutionProvider`, including `EpLibraryInternal::CreateWebGpuEp`.

There is no `EpLibrary` that accepts factory entry points directly, so a provider compiled into the host cannot reach
the plugin path.

This blocks ORT Web specifically. `cmake/onnxruntime_providers_webgpu.cmake` raises a `FATAL_ERROR` for the WebGPU
shared-module build under Emscripten, so static linking is the only way a Wasm build can ever reach the plugin
boundary.

## Goals

- Register a plugin EP that is linked into the host, reusing the existing plugin registration path after the point
  where symbol lookup would occur.
- Present statically linked plugin code with the same runtime conditions as dynamically loaded plugin code.
- Require no change to provider sources between the two linkage modes.
- Avoid new host call sites for in-tree providers.
- Keep the shared-library path byte-for-byte unchanged at runtime.

## Non-goals

- A public C API for registering static plugin EPs. Deferred until a provider is genuinely out-of-tree.
- Removal of the EP API adapters. Tracked separately.
- Removal of the direct `IExecutionProvider` WebGPU path. That is the final step of the workstream, not this change.
- Support for static plugin EPs in minimal builds. `cmake/onnxruntime_session.cmake` excludes all of
  `core/session/plugin_ep/` when `onnxruntime_MINIMAL_BUILD` is set.

## Decisions

### D1: Registration mechanism is a link-time registry in ORT core

ORT core holds a hand-written, `#if`-guarded list of statically linked plugin EP entry points, mirroring the shape of
`EpLibraryInternal::CreateInternalEps`. A new `EpLibraryStaticPlugin` accepts the entry points directly and reuses
`EpLibraryPlugin`'s factory lifecycle logic without the dynamic-library load and unload steps.

Rationale: every host — `onnxruntime_test_all`, the Python bindings, `onnxruntime.dll`, and Wasm — gets the provider
with no per-host call site.

Alternatives considered:

- CMake-generated registry using an X-macro. Rejected: added build-system machinery for generality that may never be
  needed while providers remain in-tree.
- A public `RegisterStaticExecutionProviderLibrary` C API called by each host. Rejected for now: it requires a call
  site in every statically linked host and introduces a window in which `GetEpDevices` reports no devices. Revisit
  when an out-of-tree provider needs static linking.

### D2: Entry point names are macro-parameterized

`onnxruntime/core/providers/webgpu/ep/api.cc` is parameterized by `ORT_PLUGIN_EP_ENTRY_POINT_PREFIX`.

The shared build must continue to export the unprefixed `CreateEpFactories` and `ReleaseEpFactory`, because
`EpLibraryPlugin::Load` resolves those exact names. The static build emits prefixed variants, for example
`WebGpu_CreateEpFactories`, so that multiple static plugins can coexist in one binary.

### D3: No new EP library ABI hook for teardown

All teardown remains in `ReleaseEpFactory`. Process-global shutdown that belongs to the host — currently
`google::protobuf::ShutdownProtobufLibrary()` — is compiled out under `ORT_PLUGIN_EP_OWNS_PROCESS_GLOBALS`.

Rationale: a new optional entry point cannot be relied upon, because an older ORT would load a newer plugin and
silently ignore it. Behavior that varies with the host version is worse than a compile-time decision.

The legacy static WebGPU cleanup block in `onnxruntime/core/session/ort_env.cc` is already guarded by
`defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)`. The static plugin configuration defines
`ORT_USE_EP_API_ADAPTERS`, so the block compiles out on its own and needs no edit. It must be kept for the default
internal-EP WebGPU build.

Consequence to test for: static linking has no library unload, so provider process-global state survives
unregistration. Registering, running, unregistering, re-registering, and running again is the primary regression case.

### D4: One new durable build option

- `onnxruntime_USE_EP_API_ADAPTERS` is unchanged. It is transitional and is retired when WebGPU moves fully onto the
  public plugin API.
- `onnxruntime_WEBGPU_STATIC_PLUGIN` is new and durable. It selects linkage into the host rather than a loadable
  module.
- `onnxruntime_WEBGPU_LINKED_INTO_HOST` is derived and used at the linkage sites.

`onnxruntime_USE_EP_API_ADAPTERS` currently conflates three meanings: compiling against the adapters, building a
separate module, and registering by path in tests. The static plugin configuration answers the first yes and the
other two no, so only the linkage and test sites move to the derived option. The global `add_compile_definitions` in
`cmake/CMakeLists.txt` stays as-is, and no provider sources change.

`cmake/onnxruntime_providers_webgpu.cmake` gains a third arm. The Emscripten and `onnxruntime_BUILD_CACHE`
`FATAL_ERROR`s narrow to the shared-module case, which is what unblocks ORT Web.

`build.py` gains `--use_webgpu static_plugin`.

`onnxruntime_WEBGPU_STATIC_PLUGIN` validates its prerequisites at configure time and fails with `FATAL_ERROR` if
they are not met: it requires `onnxruntime_USE_WEBGPU` and `onnxruntime_USE_EP_API_ADAPTERS`, and it rejects a
minimal build. The last one is a temporary guard rather than a permanent restriction. A minimal build excludes
`core/session/plugin_ep` from `onnxruntime_session_srcs`, and `Environment::CreateAndRegisterStaticPluginEps` is
compiled out with it, but `cmake/onnxruntime_providers_webgpu.cmake` still builds and links the provider. The
combination would therefore produce a binary containing the WebGPU EP that never registers it, with nothing
reporting the problem at configure, build or run time. Allowing minimal builds to use a static plugin EP is
worthwhile and needs the registration path to be available there first; the exclusion is currently justified by
provider-bridge dependencies, and only two of the files under `plugin_ep` are provider-bridge, so the subset may be
separable. Until then a loud configure error is preferable to a silently EP-less binary.

The CUDA plugin is unaffected: it is gated by `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN` and sets
`ORT_USE_EP_API_ADAPTERS` as a private target compile definition, never referencing the CMake option. CUDA's split
between role (`BUILD_CUDA_EP_AS_PLUGIN`) and mechanism (`ORT_USE_EP_API_ADAPTERS`) is the in-tree template for
WebGPU's eventual cleanup.

### D5: Test infrastructure treats the library path as optional

In `onnxruntime/test/unittest_util/test_dynamic_plugin_ep.cc`, `InitializationConfig::ep_library_path` becomes
optional. When absent, library registration is skipped because ORT core has already performed it, and the RAII
registration handle is left empty. The existing handle deleter already tolerates an empty handle. Device selection,
de-duplication, and factory creation are unchanged.

Virtual devices are enabled through the `allow_virtual_devices` environment configuration entry
(`kOrtEnvAllowVirtualDevices`) supplied at environment creation, rather than the `.virtual` registration-name suffix.
The suffix is unavailable because ORT core chooses the registration name for statically linked providers.

The `dynamic_plugin_ep_infra` naming becomes inaccurate. Renaming is deferred to a separate mechanical change.

### D6: Static plugin EPs are registered after the environment is published

This is the subtlest decision and the one that constrains the others.

**Problem.** The natural insertion point is next to `CreateAndRegisterInternalEps` inside `Environment::Initialize`.
Registration enumerates devices, which calls provider code:

```
OrtEnv::GetOrCreateInstance()
  lock(m_)                                          non-recursive
  Environment::Create() -> Initialize()
    Environment::CreateAndRegisterStaticPluginEps()
      Environment::RegisterExecutionProviderLibrary()
        EpInfo::Create -> factory GetSupportedDevices()
          webgpu::ep::Factory::GetSupportedDevices()   provider code
            Api().ep.GetEnvConfigEntries()
              OrtEnv::TryGetInstance()
                lock(m_)                            self-deadlock
```

`Environment::CreateAndRegisterInternalEps` already documents this hazard, and the internal WebGPU factory avoids it
by capturing `allow_virtual_devices` at construction. A plugin factory cannot use that workaround, because the public
API is all it has.

The problem is not confined to one function. Plugin EPs receive the full `OrtApi`, so any API that resolves the
environment is affected, including `CreateEnv`, which re-enters `GetOrCreateInstance` and deadlocks on the same mutex.

**Decision.** `OrtEnv::GetOrCreateInstance` publishes `p_instance_` and takes its own reference *before* invoking
`Environment::CreateAndRegisterStaticPluginEps`, and `OrtEnv::m_` becomes a `std::recursive_mutex`. Registration
therefore runs against a fully constructed and published environment.

Properties:

- The entire `OrtApi` behaves normally for provider code. `TryGetInstance` finds a published instance, and a
  re-entrant `CreateEnv` acquires the recursive lock and increments the reference count.
- No new contract for provider authors, and nothing for a future API author to remember.
- No race. Other threads block on `m_` for the duration, which is already true across `Environment::Create`.
- The reference count cannot reach zero mid-construction, because the creating thread's reference is taken first.
- Provider sources need no conditional compilation for the two linkage modes.

Costs and limits:

- Failure during registration requires explicit teardown of the just-published instance.
- A recursive mutex is normally undesirable. Here it states the actual invariant: the environment creation path can
  legitimately re-enter the environment accessor on the same thread.
- Callers of `onnxruntime::Environment::Create` do not get static plugin EPs. In the tree this is one production call
  site plus tests and orttraining sample binaries.
- Provider code that starts a thread which calls `CreateEnv` and then joins it will deadlock. This is already true of
  any code running inside `Environment::Create`.

The call to `Environment::CreateAndRegisterStaticPluginEps` therefore lives in `ort_env.cc`. Static plugin
registration is a process-singleton concern tied to the `OrtEnv` lifetime, so the singleton wrapper is its correct
home.

Alternatives considered:

- Thread-local pointer to the environment under construction, consulted by `GetEnvConfigEntries`. Rejected: it
  addresses one function, while the reachable surface is the whole `OrtApi`.
- Defer device enumeration until after environment creation. Rejected: lazy initialization moves registration
  failures from environment creation to the first `GetEpDevices` call, which is a worse place to report them.
- Register outside the lock after publishing. Rejected: another thread can observe a published environment whose
  static EPs are not yet registered, and avoiding that requires a second lock and a completion barrier.

`onnxruntime::Environment` remains directly constructible, and tests rely on that. `OrtEnv::GetOrCreateInstance`
returns any existing instance and ignores the logging manager, threading options, and configuration entries passed by
later callers, so a test needing specific threading or logging configuration must bypass it. Those tests are
deliberately opting out of process-singleton semantics and should not receive static plugin EPs.

### D7: The statically linked build does not use manual C++ API initialization

`include/onnxruntime/ep/api.h` force-enables `ORT_API_MANUAL_INIT` around its include of `onnxruntime_cxx_api.h`, and
`onnxruntime::ep::ApiInit` calls `Ort::InitApi(ort_api)`. That is required for a plugin EP shared library, which must
not call `OrtGetApiBase()` itself.

`onnxruntime_cxx_api.h` emits `#pragma detect_mismatch("ORT_API_MANUAL_INIT", ...)` on MSVC, so every translation
unit linked into one binary must agree. The statically linked plugin EP is linked with ORT core and with test code,
neither of which uses manual initialization, so forcing it on produces `LNK2038` for every EP object file.

The build therefore defines `ORT_PLUGIN_EP_STATICALLY_LINKED` on the statically linked plugin EP target. Under that
macro `ep/api.h` includes `onnxruntime_cxx_api.h` unmodified and `ApiInit` skips `Ort::InitApi`. The C++ API then
default-initializes from `OrtGetApiBase()->GetApi(ORT_API_VERSION)`, which resolves in-process and yields the same
`OrtApi` that `ApiInit` would have installed, because the EP and ORT are the same binary and therefore the same
version. `onnxruntime::ep::ApiPtrs` is still populated from the `OrtApiBase*` that ORT passes to `CreateEpFactories`,
so the EP's own API access is unchanged.

The macro is generic rather than WebGPU-specific because `ep/api.h` is shared with the CUDA plugin EP.

## Implementation order

1. `EpLibraryStaticPlugin`, factoring the shared factory lifecycle out of `EpLibraryPlugin`.
2. The static registry and `Environment::CreateAndRegisterStaticPluginEps`, with the `ort_env.cc`
   publish-then-register change.
3. `ep/api.cc` entry point prefixing and host-owned process-global guards.
4. Build option and `build.py` changes.
5. Test infrastructure changes.
6. `ORT_PLUGIN_EP_STATICALLY_LINKED` in `include/onnxruntime/ep/api.h`, per D7.
7. Applying the WebGPU plugin EP test definitions to `onnxruntime_provider_test` as well as `onnxruntime_test_all`,
   via a shared `onnxruntime_set_webgpu_plugin_ep_test_definitions` cmake function. `onnxruntime_provider_test` holds
   the operator tests, so without this the plugin path has no operator test coverage. It also needs
   `ORT_UNIT_TEST_HAS_WEBGPU_STATIC_PLUGIN_EP` in its own right, because that macro is what enables the
   `allow_virtual_devices` environment configuration entry at environment creation in `test_main.cc`.

## Follow-up cleanup

`Environment::CreateAndRegisterInternalEps` stays where it is, at the end of `Environment::Initialize`. Internal EP
factories are ORT-core code that does not re-enter the `OrtEnv` singleton, so they are not subject to the constraint
in D6. Moving them would also remove EP devices from every directly constructed `Environment`, and `InferenceSession`
reads `Environment::GetOrtEpDevices` directly. For static plugin EPs the resulting gap is unimplemented new
functionality; for internal EPs it would be a silent regression.

The `allow_virtual_devices` parameter threaded through `EpLibraryInternal::CreateInternalEps` exists only for the
internal WebGPU EP factory, which cannot query the environment at `GetSupportedDevices` time. The comment in
`CreateAndRegisterInternalEps` documents that constraint. Both can be deleted once WebGPU is no longer an internal EP:
the remaining internal EPs are CPU, kept deliberately as a special case, and DML, which is no longer maintained.
Neither takes the parameter. In a build that defines `ORT_USE_EP_API_ADAPTERS` this is already true, since the
internal WebGPU EP is compiled out and the parameter reaches `ORT_UNUSED_PARAMETER`.

This does not affect the `.virtual` registration-name suffix or the `allow_virtual_devices` environment
configuration entry, which remain in use for dynamically registered plugin libraries.

## Risks and open questions

- Whether adapter-compiled WebGPU objects link into ORT core without duplicate-symbol or ODR problems. One real
  instance was found and resolved: the `ORT_API_MANUAL_INIT` `detect_mismatch` guard, see D7. Otherwise resolved:
  `onnxruntime_test_all` and `onnxruntime_provider_test` both link cleanly against the statically linked EP.
- Provider process-global state surviving unregistration, since static linking has no unload event. Covered by the
  re-registration regression case in D3. Not yet exercised: `OrtEnv` is a reference-counted process singleton and the
  unit test main holds a reference for the lifetime of the process, so a second `Environment` construction (and
  therefore a second `CreateEpFactories` call) is not reachable from the existing test binaries. This needs either a
  dedicated single-test process or a test that fully releases and recreates the environment.
- Whether any shipped ORT Web configuration uses an extended-minimal build, which excludes plugin EP infrastructure
  entirely.
- Whether any `std::condition_variable` is paired with `OrtEnv::m_`, which would require `condition_variable_any`
  under a recursive mutex.
- The existing rejection of `onnxruntime_BUILD_DAWN_SHARED_LIBRARY` together with the adapters. Initially left
  rejecting both plugin kinds.
- Binary size and dead-code elimination for statically linked providers, per the workstream completion criteria.

## Validation status

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

### Known test coverage difference on the plugin path

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

### CI coverage

No pre-existing CI leg built this configuration, since both new CMake options default to `OFF`. Two jobs were added:

- `webgpu_static_plugin_build_x64_RelWithDebInfo` in `.github/workflows/windows_webgpu.yml`, which builds and runs the
  tests on a GPU-equipped pool. This closes the operator execution gap above. Note that it must not set
  `onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON`, which is incompatible with `onnxruntime_USE_EP_API_ADAPTERS`.
- `build-linux-webgpu-static-plugin-x64-release` in `.github/workflows/linux_webgpu.yml`, build only, mirroring its
  sibling job. This is the only GCC coverage of the EP adapters, and the only coverage of the `onnxruntime_python.cmake`
  change, hence `--build_wheel` is kept.

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

### Emscripten and ORT Web

Making the statically linked WebGPU EP reachable through the plugin path is what motivates this design, so the
Emscripten configuration was built before landing the change: `--build_wasm --enable_wasm_simd
--enable_wasm_threads --use_webgpu static_plugin --use_webnn --target onnxruntime_webassembly`. It configures,
compiles and links cleanly, producing `ort-wasm-simd-threaded.asyncify.{mjs,wasm}`.

`static_plugin` is in fact the only plugin kind Emscripten supports — `cmake/onnxruntime_providers_webgpu.cmake`
raises a `FATAL_ERROR` for `shared_lib` there, since there is no runtime library to load.

No CI leg covers this yet. The WASM WebGPU jobs pass a bare `--use_webgpu`, which defaults to `static_lib`
(`tools/ci_build/build_args.py` uses `const="static_lib"`), so they still build the internal EP. This change makes
the configuration available to ORT Web; switching ORT Web over to it is separate work, and has prerequisites that
this change does not address. See [ORT Web migration](#ort-web-migration).

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
[ORT Web migration](#ort-web-migration).

A practical build note: the WASM build needs `node` on `PATH` — `node_helper.cmake` does a hard `find_program` and
fails configure without it. emsdk bundles a suitable one under `cmake/external/emsdk/node/<version>/bin`.

## ORT Web migration

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

4. Flip the ORT Web WASM builds to `static_plugin` and add a CI leg. The two builds are
   `Build (simd + threads + WebGPU experimental)` and its JSPI sibling in
   `.github/workflows/linux-wasm-ci-build-and-test-workflow.yml`, both of which pass a bare `--use_webgpu` and so
   get `static_lib` from `build_args.py`'s `const="static_lib"`. Neither uses `--minimal_build`, so the
   minimal-build `FATAL_ERROR` guard does not stand in the way of this migration. No pipeline builds WASM with
   `static_plugin` today, so without a leg this path stays validated at the symbol level only.

   *Prototype note:* the `js/web` test runner cannot currently exercise this configuration end to end, because
   `script/test-runner-cli.ts` spawns `script/build` with only `--bundle-mode` and does not forward `--webgpu-ep`.
   Setting `npm_config_webgpu_ep` in the environment works as a stopgap, since `script/build.ts` reads it, but a CI
   leg should forward the flag properly. Note also that without `--webgpu-ep` the `webgpu` backend silently routes
   through JSEP instead, so a leg that omits it would pass while testing nothing relevant.

### Interaction with the ORT Web reduced-size build options

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

#### Measured size comparison

Both modes were then built locally with the exact CI flag set (`Release`, SIMD + threads, asyncify, WebNN on,
`--disable_rtti`, `--enable_wasm_api_exception_catching`, and the full reduced-size set above) and the artifacts
compared:

| Artifact | `static_lib` | `static_plugin` | Delta |
| --- | ---: | ---: | ---: |
| `ort-wasm-simd-threaded.asyncify.wasm` | 26,550,879 B | 26,488,911 B | **−61,968 B (−0.23%)** |
| `ort-wasm-simd-threaded.asyncify.mjs` | 53,249 B | 53,249 B | 0 |

(Both binaries rebuilt after merging `main`. An earlier pre-merge pair measured −52,978 B / −0.20%; the merge added
kernels to both builds and did not change the relationship.)

**There is no size regression** — `static_plugin` is marginally smaller. Both configurations compiled cleanly with
every reduced-size flag, which empirically confirms the analysis above. The plugin build plausibly comes out ahead
because it drops the internal `IExecutionProvider` / `KernelRegistry` glue and the built-in `WebGpuEpFactory` in
`ep_library_internal.cc` in favour of the header-only adapters, but that attribution is not separately measured.
For reference, the same `static_plugin` configuration without any reduced-size flags is 40.9 MB, so those flags are
worth ~15 MB and remain essential regardless of EP mode.

#### `--enable_wasm_api_exception_catching` and the EP API adapters

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

#### Measured performance comparison

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
  one. Amortized over ~300 nodes it is on the order of 3 µs per node.
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

> Should static factories be registered before environment creation or through environment construction options?

Neither. They are registered by ORT core during `OrtEnv` creation, after the environment is constructed and
published. See D6.
