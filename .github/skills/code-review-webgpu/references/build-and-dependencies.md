# Builds and dependencies

**Owns:** Dawn rolls, native/WASM/plugin builds, Dawn/DXC pins, patches/manifests,
staging, packaging, and update automation.
**Update when:** Dependency coordinates, flags/targets, exports, artifacts, or
supported deployment configurations change.
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
