# Updating Dawn

Run `update_dawn.py` from any directory to prepare a Dawn roll for ONNX Runtime.
It uses the active pin in `cmake/deps.txt` and the patch order in
`cmake/external/onnxruntime_external_deps.cmake`. The script, helpers, and tests
are owned by the WebGPU provider so they can move with it to a standalone repo.

## Prepare a roll

Prerequisites: Python 3.10+, Git, GNU `patch`, and network access to GitHub.
On Windows, Git for Windows supplies `patch.exe` in its `usr/bin` directory;
add that directory to `PATH` or pass `--patch-executable PATH`.

```powershell
python plugin-ep-webgpu/tools/dawn/update_dawn.py --revision v20261013.193955
```

The default mode changes no tracked files. It writes `report.md`, `report.json`, and
`changes.patch` under `build/dawn-upgrade/`. Review these files, resolve any
reported compatibility decisions, then apply the prepared roll:

```powershell
python plugin-ep-webgpu/tools/dawn/update_dawn.py --revision v20261013.193955 --apply
```

`--revision` accepts an exact tag or a full commit hash. Omitting it checks the
current pin. Downloads are cached under `build/dawn-upgrade/cache/`; cached
archives are checked for corruption and their DEPS must match the resolved
commit. Tags are resolved again after downloading to detect a moved tag.

Use `--repo-root PATH` to target another checkout, `--cache-dir PATH` to share
downloads, and `--output-dir PATH` to retain separate reports for different runs.
Normal builds continue using CMake's existing download and patch mechanisms.

## What the script does

1. Resolves Dawn's revision, downloads the exact archive, computes its SHA1,
   and prepares the corresponding `cmake/deps.txt` entry. Git commit hashes and
   archive checksums are handled separately.
2. Parses pinned DEPS and the dependency fetcher's `required_submodules` without
   executing either file. Reports added, removed, and changed Git dependencies,
   conditions, fetched paths, and unregistered entries.
3. Reuses the manifest updater in `cgmanifests/webgpu/` to update Git commits,
   renamed paths, and Dawn roots while preserving reviewed classifications.
   Obsolete development registrations are removed only when their identities
   are verified in the manifest's previous Dawn DEPS and absent from the new
   DEPS. Removed runtime entries and new/unresolved fetched dependencies block
   application until reviewed. Nested fetched paths not declared directly in
   Dawn DEPS also require review; they are never silently omitted.
4. Updates the Linux WebGPU Dockerfile's SwiftShader pin from the same DEPS.
5. Checks the active patch stack in build order against a temporary pristine
   source tree. It refreshes stale offsets/context, retaining each patch's edits.
   GNU patch context fuzz up to two lines is allowed and recorded for review;
   refreshed patches subsequently apply without fuzz or offsets. Conflicts or
   already-applied patches need manual resolution. The report includes the
   final patched-file hashes for comparing patch stacks.
6. Writes a review report with the resolved revisions, archive checksum,
   dependency changes, manifest drift, patch results, and unresolved decisions.
   Build/package verification uses the existing build and test workflows.

All edits are prepared before application. A blocker prevents all tracked-file
writes. The script checks for concurrent input changes and restores files if
an ordinary filesystem error interrupts application. It does not create Git
commits, push, open PRs, run CI, or publish packages.

## Scope

The script automates recurring metadata and patch maintenance. It does not
rewrite runtime code, package DLL lists, or build options. The one-time DXIL
removal in [#32686](https://github.com/microsoft/onnxruntime/pull/32686) is already
complete upstream and is intentionally outside this tool.

The disabled Dawn vcpkg port is explicitly excluded: its older pin and patch
stack need separate maintenance before that path is enabled again. Its
integration is checked so re-enabling it cannot silently bypass upgrade checks.
The redistributed DXC release and optional Agility SDK are independent pins;
neither is changed by this command.

## Verify the result

```powershell
python plugin-ep-webgpu/tools/dawn/update_dawn.py --check
python cgmanifests/webgpu/validate_webgpu_cgmanifest.py
```

Exit codes: `0` means the requested planning/application/check completed,
`1` means `--check` found required edits, and `2` means a blocker or error.
Successful metadata/patch checks do not establish runtime compatibility.

Before submitting a roll:

- Review patch context changes and dependency classification/condition changes.
  The complete DEPS inventory is larger than the fetched or shipped dependency
  set; newly declared optional dependencies are reported, not automatically
  classified as runtime components.
- Run the relevant Windows native, plugin, external-Dawn and minimal build
  profiles in `.github/workflows/windows_webgpu.yml`. Shared Dawn and a shared
  plugin EP are different configurations.
- Run macOS native tests and Linux runtime tests as applicable. The current
  Linux native PR workflow builds but has its runtime-test job disabled.
- Run actual WebGPU EP browser tests from `web.yml`, including the affected
  browser when changing an Emscripten/browser patch.
- Build and smoke-test final packages using the plugin packaging/test pipelines.

A successful build does not prove that every relevant test executed. Source API changes,
new upstream CMake contracts, package policy, and GPU regressions still need
engineering review. These checks can also be invoked by CI; this change does
not add or dispatch a workflow.

## Test the upgrader

```powershell
python -m pytest plugin-ep-webgpu/tools/dawn/test_update_dawn.py cgmanifests/webgpu/test_update_webgpu_cgmanifest.py -q
```

Tests use temporary files and mocked downloads, with real GNU patch for patch
ordering and refresh tests. They cover dry runs, checks, idempotence, corrupt
archives, moved tags, metadata preservation, removal policy, distinct revisions
of the same repository, patch conflicts, concurrent edits, and failed-write
recovery. Package files are left untouched.

## Relationship to component inventories

The roll command updates the checked-in, reviewed manifest. It preserves
runtime/development classifications and the independently redistributed DXC
release registration. It reports newly fetched dependencies for classification
rather than guessing whether they ship in a package. Matching a registration
requires the repository identity **and** revision; a second checkout of the same
repository at a different revision is a separate input.

The [build-input inventory generator](../../../../cgmanifests/webgpu/BUILD-INVENTORY.md)
serves a different purpose: after CMake fetches dependencies, it records their
actual checkout commits for component scanning. It does not replace the roll
command or the reviewed manifest. Neither inventory alone describes a package's
final linked/redistributed contents.

## Upgrade examples used to validate the tool

- [#32686](https://github.com/microsoft/onnxruntime/pull/32686), Dawn
  `v20260916.214021`: archive checksum, dependency commits and roots, SwiftShader,
  and active patch stack. Its one-time package migration and manual explanatory
  comments are not automated.
- [#33390](https://github.com/microsoft/onnxruntime/pull/33390), Dawn
  `v20261013.193955`: the routine `cmake/deps.txt` and manifest updates. A roll
  also checks SwiftShader and retained patches even if they were not edited in
  the example PR.

A routine roll can be prepared with the commands above, reviewed through
`report.md` and `changes.patch`, applied with `--apply`, then checked again with
`--check`. A clean check means the metadata and patch plan is stable; API and
runtime compatibility still need the build/test steps above.
