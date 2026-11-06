# Builds and dependencies

**Owns:** Dawn rolls, native/WASM/plugin build boundaries, Dawn/DXC pins, retained patches,
component manifests, staged artifacts, and dependency-update automation.
**Update when:** Dependency coordinates, build flags/targets, export sets,
packaging inputs, or supported deployment configurations change.

## Dawn roll workflow

1. Record the old and new exact Dawn commits, tag/archive checksum, and reason
   for the roll. Inspect the upstream change range for ORT-facing WebGPU/proc
   APIs, callback/future behavior, Tint/WGSL validation, features, limits, and
   backend changes. Select runtime, device, shader, or numerical checks for
   affected behavior; a metadata-only ORT diff can still change execution.
2. For authoring, use the repository's [Dawn update workflow](../../../../plugin-ep-webgpu/tools/dawn/README.md)
   with the intended `--revision`. Review its generated report and final diff.
   It updates the source pin/checksum, dependency registrations, Linux test
   SwiftShader pin, and active patches, then validates metadata and patches.
   Successful tool validation does not establish compilation or GPU behavior.
3. Audit the exact new DEPS and fetcher's required dependencies using the
   [component manifest policy](../../../../cgmanifests/webgpu/README.md).
   Classify new/removed components against actual build options and shipped
   contents. Preserve the independently redistributed DXC release identity;
   the updater does not roll DXC, Agility SDK, or package DLL lists. Revisit
   dormant integration pins if their build path becomes active.
4. Apply the active patch stack in CMake order to pristine pinned source.
   Explain retained, refreshed, or removed patches; verify an upstream fix
   before dropping its workaround. Review patch behavior as well as clean
   application, including backend guards and security/build flags.
5. Follow the build, external-host, and packaging checks below. Exercise
   changed shaders with full validation and actual GPU execution on affected
   backends; include capture, callbacks, or concurrent-session tests when the
   roll changes those contracts. Recheck browser/WASM compatibility for changed
   shared API/WGSL use. Record unavailable targets and tests as gaps using
   [validation](validation.md), with the tested Dawn and compiler identities.

## Build and artifact checks

- Trace supported native Dawn backends (D3D12/Vulkan/Metal), browser WASM,
  built-in provider, shared plugin, and static plugin as affected. Inspect
  CMake branches and compile definitions; one successful configuration does
  not prove another. New ORT dependencies must reach the plugin source set too.
  The Windows WebGPU workflow compiles the plugin target; recheck current jobs.
- Pin exact Dawn/DXC sources and redistributed binaries independently. Follow
  `cmake/deps.txt`, external CMake, `cmake/patches/dawn/`, component manifests,
  plugin staging, Node, Python wheels, and NuGet consumers of changed artifacts.
  A provider build is not packaging evidence.
- Apply each retained patch to the exact new source; verify hunk integrity and
  backend guards. Dawn patches need evidence from every backend they affect.
  Generate component registrations from the new DEPS, including added/removed
  children and each repository/hash pair. A root-only pin check is insufficient.
- For removed DLLs, distinguish static imports, delay loads, explicit loads,
  and optional internal loads. Test a clean staged package and shader compile,
  D3D12 pipeline creation, and GPU execution as required. Record loaded compiler
  paths and unexpected loads; do not disable shader validation to pass a probe.
- Source-built and host-provided/external Dawn need compatible proc tables,
  generated headers, and version identity. A linked library or successful DLL
  load does not prove runtime compatibility. Keep intentionally unsupported
  combinations explicit, including accelerated-loader/platform restrictions.
- For dependency-update tools, validate archives/paths and artifact hashes.
  Multi-file changes must recover partial writes and attempt every restoration,
  retaining original and rollback errors. Test persistent failures, not only
  mocks that allow the rollback write. Keep generated/temporary outputs out of
  the target-relative PR diff.

Review evidence: [#32686 packaging](https://github.com/microsoft/onnxruntime/pull/32686#discussion_r4043364266),
[patch integrity](https://github.com/microsoft/onnxruntime/pull/32686#discussion_r4043364307),
[component revisions](https://github.com/microsoft/onnxruntime/pull/32686#discussion_r4058704693),
and [#33396 rollback](https://github.com/microsoft/onnxruntime/pull/33396#discussion_r4321149690).
Use [validation](validation.md) to distinguish compilation, mock packaging,
artifact loading, shader compilation, and real GPU execution evidence.
