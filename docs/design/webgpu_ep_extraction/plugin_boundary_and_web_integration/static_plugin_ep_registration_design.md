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

Not yet verified: operator execution. The development machine is a virtual machine with no real GPU, so only the
virtual WebGPU device is surfaced and non-compile-only sessions are rejected up front. This was confirmed to be a
property of the host rather than of static linking by building the shared library plugin configuration
(`--use_webgpu shared_lib`) on the same host and running the same tests: it passes and fails exactly the same set
(`WebGpuVirtualDeviceCompileOnlyEndToEnd` and `WebGpuVirtualDeviceRejectedWithoutCompileOnly` pass;
`WebGpuCompileOnlySkipsFinalization` and `TestStrictShapeInference` fail with the identical virtual-device rejection
message). Operator coverage on either plugin path requires a host with a GPU.

The shared library plugin configuration was also re-verified after the D7 change, since `include/onnxruntime/ep/api.h`
is shared with the CUDA plugin EP. It builds, and `onnxruntime_providers_webgpu.dll` still exports exactly the
unprefixed `CreateEpFactories` and `ReleaseEpFactory` required by `EpLibraryPlugin::Load`, per D2.

## Resolution of workstream open questions

> Should static factories be registered before environment creation or through environment construction options?

Neither. They are registered by ORT core during `OrtEnv` creation, after the environment is constructed and
published. See D6.
