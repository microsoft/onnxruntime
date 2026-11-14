# Platform support and release

**Owns:** Which devices/environments can use the EP and how it is integrated/shipped:
options, device features, host/plugin ABI, native/WASM builds, Dawn/DXC, and packages.
**Update when:** Device selection, enabled features, compatibility, dependencies,
build inputs, or shipped artifacts change. Operator acceptance belongs to
[computation](computation.md#registration); live resource ownership belongs to
[execution](execution.md). Review source and existing CI only; do not run build/update tools.

## Devices and options

- Trace public options to effective behavior in every affected binding/build.
  Reject malformed/trailing/out-of-range values and unsupported combinations;
  preserve absent/explicit values and defaults. JS types do not prove Node or
  browser/WASM forwarding: require documented-API session-creation coverage.
- Adapter indices depend on enumeration configuration. Compare the explicit
  (`adapterIndex`, `powerPreference`, `dawnBackendType`) tuple on context reuse.
  Nonzero `deviceId` may select a custom instance/device pair, not a physical
  adapter index. Power preference proves neither GPU class nor hardware execution.
- Gate `shader-f16`, subgroups, subgroup matrix/IDs, and subgroup-size-control
  on **enabled device** features, including supplied devices. Reconcile
  enable/disable lists, dependencies, limits, and conflicts; adapter support alone
  is insufficient.
- Separate algorithm constraints from vendor restrictions; retain/test generic
  and ineligible fallbacks. Measure actual model-layer eligibility. Native-only
  configuration tables can break WASM despite shared shader sources.
- Device-free compile-only/virtual modes must avoid forbidden enumeration or
  allocation and reject virtual-device inference clearly. Software-adapter
  advertisement does not prove hardware execution.
- Parser/compile-only tests may bypass device creation/reuse. Add guarded runtime
  coverage for affected custom devices, unsupported features, selectors, and
  reuse rejection.

Inspect `webgpu_provider_factory.cc`, `webgpu_context.cc`,
`webgpu_provider_options.h`, and `ep/factory.cc`. Session-varying codegen options
also need [cache identity](computation.md#cache-identity).

## Hardware workarounds

- Scope vendor workarounds to device/backend/feature evidence; retain correctness
  rationale/fallback and test unaffected paths. Old failures are not universal
  current restrictions.

## Plugin contracts

- Distinguish release version, build SDK/API bounds, minimum host, and runtime
  negotiation. Verify API and patch-release predicates in `ep/runtime_compatibility.*`;
  one fixed release line does not cover intervening lines. Preserve conservative
  compatibility/overrides; mocks do not prove multi-runtime GPU support.
- Initialize APIs before wrappers or exception-to-status macros; failed
  negotiation still needs supported status reporting. Audit transitive
  allocation/Free/destruction under `noexcept` C callbacks; contain exceptions
  using boundary conventions and preserve cleanup.

- Distinguish `OrtMemType`, `OrtDeviceMemoryType`, and memory-device identity.
  Verify vendor/device IDs, default/read-only allocators, session/Env ownership,
  streams/notifications, and release order against the host ABI.
- Shared plugins need exact exports/runtime dependencies; static plugins need
  prefixed entry points/API initialization. Check factory registration/removal
  and context/logger/library teardown.
- New internal dependencies must reach `plugin-ep-webgpu/paths.txt` and EP
  adapters. Inspect other adapter consumers (including CUDA) and affected native,
  shared/static plugin, and WASM integration via [build checks](platform.md).

Inspect `ep/{api,ep,factory,allocator,sync_stream}.cc`, public EP APIs/adapters,
and session `ep_factory_webgpu.*`.

Review source/configuration and existing CI evidence only; do not invoke the
updater, builds, packaging, or CI. The workflow below describes authoring and
the evidence a reviewer assesses.

## Dawn roll workflow

1. Record exact old/new Dawn commits, tag/archive checksum, and reason. Inspect
   upstream API/proc, callback/future, Tint/WGSL, feature/limit, and backend changes;
   select affected runtime/device/shader/numeric checks even for metadata-only diffs.
2. Author with the [Dawn updater](../../../../plugin-ep-webgpu/tools/dawn/README.md)
   and intended `--revision`. Review report/diff: source pin/checksum,
   registrations, Linux test SwiftShader pin, and active patches. Metadata/patch
   validation does not establish compilation or GPU behavior.
3. Audit exact new DEPS/fetcher dependencies under the
   [manifest policy](../../../../cgmanifests/webgpu/README.md). Classify additions/
   removals against build options and shipped contents. DXC redistribution,
   Agility SDK, and package DLL lists have independent pins; the updater does
   not roll them. Recheck dormant pins when their paths become active.
4. Apply active patches in CMake order to pristine pinned source. Explain
   retention/refresh/removal; verify upstream fixes before dropping workarounds.
   Check behavior, backend guards, and security/build flags, not just application.
5. Assess artifact checks and affected-backend CI shader results with full
   validation. Include capture/callback/concurrency evidence when changed,
   browser/WASM compatibility for shared APIs/WGSL, and tested Dawn/compiler
   identities. Record unavailable targets via [validation](validation.md).

## Build and artifact checks

- Inspect affected D3D12/Vulkan/Metal, browser WASM, built-in, shared/static plugin
  CMake paths/definitions and source sets. One configuration proves only itself;
  recheck current Windows WebGPU plugin compilation jobs.
- Trace independent source/binary pins through `cmake/deps.txt`, external CMake,
  `cmake/patches/dawn/`, manifests, plugin staging, Node, wheels, and NuGet.
  A provider build does not prove packaging.
- Patch/manifest edits outside rolls still need exact-source hunk/guard checks,
  evidence from affected backends, and every added/removed DEPS child repository/hash;
  a root pin alone is insufficient.
- Before removing DLLs, distinguish static imports, delay/explicit/optional loads.
  Require evidence from clean staged packages through shader compilation, D3D12 pipeline creation,
  and GPU execution as required. Record loaded compiler paths/unexpected loads;
  keep shader validation enabled.
- Source-built and host/external Dawn need compatible proc tables, generated
  headers, and version identity; linking/loading is insufficient. State unsupported
  combinations, including accelerated-loader/platform restrictions.
- Update tools must validate archives/paths/hashes, recover partial multi-file
  writes, attempt every restoration, and retain original plus rollback errors.
  Check persistent-failure tests beyond rollback-friendly mocks; exclude temporary/
  generated outputs from the target-relative PR diff.

Distinguish compilation, mock packaging, loading, shader compilation, and real
GPU execution using [validation](validation.md).
