# Updating Dawn

From the repository root, run one command to update Dawn to the newest dated
release and validate the resulting metadata and patches:

```powershell
python plugin-ep-webgpu/tools/dawn/update_dawn.py
```

This is equivalent to `--revision latest`. To select a particular release or
full Git commit instead:

```powershell
python plugin-ep-webgpu/tools/dawn/update_dawn.py --revision v20261016.190333
```

The command applies the update directly. There are no separate dry-run, apply,
or check modes, and no separate manifest-validation command is required.
Repeated runs for the same revision succeed without further edits.

## Prerequisites

Use Python 3.10+, Git, GNU `patch`, and network access to GitHub. On Windows,
Git for Windows supplies `patch.exe` in its `usr/bin` directory; add that
directory to `PATH` or pass `--patch-executable PATH`.

Start from a branch containing this tooling and keep unrelated edits separate
so the generated diff is easy to review.

## What happens automatically

1. Resolve the requested revision and download its exact archive. `latest`
   selects the greatest timestamp among Git tags matching `vYYYYMMDD.HHMMSS`.
   Record the concrete tag and commit, compute the archive SHA1, compare archive
   DEPS against the resolved commit, and detect a tag moving during download.
2. Update `cmake/deps.txt` and the reviewed manifest's dependency revisions and
   Dawn roots. Parse DEPS and the fetcher's dependency list without executing
   them. Preserve classifications, unrelated metadata and the independently
   packaged DXC release registration.
3. Synchronize the Linux test Dockerfile's SwiftShader revision with Dawn DEPS.
   This is the explicit software Vulkan driver used by Linux package tests.
4. Apply active patches in CMake order to temporary pristine source. Refresh
   stale locations/context when possible, then verify each refreshed patch
   produces the same source with zero fuzz or offsets before writing it.
5. Apply the prepared file changes, verify the written contents, and run the
   existing manifest checks automatically against this checkout: Dawn pin,
   dependency roots, and the independently packaged DXC release metadata.

Unresolved patches, changed repository identities, and new fetched dependencies
needing classification stop the update. Verified obsolete development-only
registrations can be removed; removed runtime entries require review. Nested
fetched paths not declared directly in Dawn DEPS also require review.

The command checks that inputs have not changed while preparing the update.
Writing or final-validation errors trigger restoration of every file whose write
was attempted, including a partially written file. If a file cannot be restored,
rollback continues for the remaining files and reports all restoration errors
alongside the original failure. Review any reported paths before retrying.
It exits `0` after a successful update and validation, and nonzero on failure.

## Results

`build/dawn-upgrade/` contains `report.md`, `report.json`, and `changes.patch`.
The report records the target tag/commit, archive checksum, changed files,
dependency changes, patch results, and update status. These are records for
review; they do not require another apply step.

Downloads are cached under `build/dawn-upgrade/cache/`. Cached archives are
checked for corruption. Latest-tag discovery still uses the network on every
invocation. Use `--repo-root PATH` for another checkout, `--cache-dir PATH` for
a download cache, and `--output-dir PATH` for a separate report directory.

## Scope and build validation

The command updates source metadata and patches; it does not compile ORT, run
GPU tests, publish packages, create commits, or open PRs. Review the resulting
Git diff and use the applicable native, plugin, external-Dawn, browser, and
package build/test workflows to establish runtime compatibility. Feature-specific
GPU changes still need tests on the relevant hardware.

It does not change runtime code, package DLL lists, build options, the separately
redistributed DXC release, or the Agility SDK pin. The completed DXIL removal is
outside this tool. The dormant Dawn vcpkg port is excluded; if its integration
is re-enabled, the tool stops so its pin and patch stack can be reviewed.

Dependency classifications remain review decisions based on actual CMake options
and package contents. The complete DEPS inventory is not the shipped dependency
set. The existing manifest validator remains independently usable for other
maintenance tasks, but this command runs its checks automatically.

## Tool tests

```powershell
python -m pytest plugin-ep-webgpu/tools/dawn/test_update_dawn.py cgmanifests/webgpu/test_update_webgpu_cgmanifest.py -q
```

Tests use temporary files, mocked downloads, and real GNU patch application.
They cover automatic updates and validation, repeat runs, latest-tag discovery,
corrupt archives, moved tags, metadata preservation, patch conflicts, strict
verification of refreshed patches, concurrent edits, and rollback on write or
validation failure.

The workflow was also compared with the recurring changes in ORT PRs #32686 and
#33390 using pinned source snapshots and actual Dawn archives. One-time packaging
migrations and manually reviewed explanatory comments are outside its scope.
