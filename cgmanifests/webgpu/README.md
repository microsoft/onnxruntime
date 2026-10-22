# WebGPU Component Governance manifest

This directory contains the WebGPU-specific Component Governance manifest for ONNX Runtime. It covers Dawn and the
Dawn-derived dependency graph used when building the WebGPU Execution Provider.

The manifest is named `cgmanifest.webgpu.json`, not `cgmanifest.json`, so default whole-repository Component
Governance scans do not pick it up automatically. WebGPU packaging and NOTICE-generation pipelines should stage or copy
this file as `cgmanifest.json` in the source directory that they scan for WebGPU package notices.

## Classification policy

The Component Governance manifest schema provides a `developmentDependency` boolean, but it does not provide separate
first-class fields for runtime, build-tool, test-only, or conditional dependencies. This manifest uses:

- no `developmentDependency` field for components that are redistributed, statically linked, or otherwise part of the
  WebGPU package/runtime dependency closure;
- `developmentDependency: true` for Dawn dependencies that are only build tools, tests, disabled optional backends, or
  source inputs that current WebGPU packages do not redistribute;
- `comments` to preserve the more precise classification and Dawn `DEPS` path/condition.

If a WebGPU package starts redistributing a component currently marked as a development dependency, update that
registration and explain the packaging path in `comments` and `detectedComponentLocations`.

## Maintenance

When rolling Dawn or changing WebGPU packaging:

For a complete Dawn update, run the [Dawn update tool](../../plugin-ep-webgpu/tools/dawn/README.md):

```powershell
python plugin-ep-webgpu/tools/dawn/update_dawn.py
```

It selects the latest dated release by default, updates the archive pin/checksum, this manifest, SwiftShader,
and active patches, then validates the result automatically. Use `--revision TAG_OR_COMMIT` to choose a version.
The commands below are for manifest-only maintenance; they are not additional steps after a complete update.

1. After updating the `dawn` entry in `cmake/deps.txt`, refresh the registered Git commit hashes:

   ```powershell
   python cgmanifests/webgpu/update_webgpu_cgmanifest.py
   ```

   The updater resolves Dawn's commit or tag, downloads `DEPS` at that exact commit, and updates the Dawn root,
   registered Dawn dependency commits, and `dependencyRoots`. It recognizes Chromium mirrors of public repositories
   and updates renamed DEPS paths in comments. It preserves the reviewed classifications, other comments, public
   repository URLs, and detected locations, including the separate redistributed DXC release registration.

   This command needs network access and Git on `PATH` when Dawn is pinned to a tag. It parses DEPS without executing
   it and does not clone dependency repositories or evaluate platform conditions. A missing, changed, or ambiguous
   registered repository stops the update before writing the manifest; review that entry using the inventory below.

2. Re-audit the Dawn dependency graph for the pinned Dawn commit. To generate a JSON inventory of every Git entry in
   Dawn's DEPS (path, repository URL, commit hash, and condition), run:

   ```powershell
   python cgmanifests/webgpu/update_webgpu_cgmanifest.py --list-deps
   ```

   - Start from the Dawn commit in `cmake/deps.txt`; do not audit Dawn `main` or a different roll.
   - Inspect Dawn's `tools/fetch_dawn_dependencies.py` at that commit. For ORT's normal source-fetch path,
     `cmake/external/onnxruntime_external_deps.cmake` enables `DAWN_FETCH_DEPENDENCIES`, so the script's
     `required_submodules` list is the primary set of Dawn source dependencies fetched for the build.
   - Cross-reference each fetched submodule path with the generated inventory. Its URLs are Dawn's checkout URLs;
     use public upstream identities in this manifest, not internal mirrors.
   - Compare that fetched set against this manifest. The updater refreshes existing registrations; it does not decide
     which components belong in the manifest. Add new fetched components, review changed repository URLs, and remove
     entries that are no longer fetched or relevant unless CG/legal guidance requires keeping them. Rerun the updater
     after editing registrations.
   - Cross-check ORT's Dawn CMake options in `cmake/external/onnxruntime_external_deps.cmake` and Dawn's
     `third_party/CMakeLists.txt` before classifying a component. Components that are redistributed, statically linked,
     or otherwise part of the WebGPU package/runtime closure should not be marked as development dependencies; build
     tools, test inputs, disabled optional backends, and unfetched conditional dependencies should be marked
     `developmentDependency: true` if they remain registered.
   - Verify actual WebGPU package contents, especially platform-specific artifacts. For example, the Windows WebGPU
     plugin pipeline downloads and redistributes DXC DLLs separately from Dawn's `third_party/dxc` source dependency, so
     both the Dawn build-input registration and the redistributed DXC release registration may need review.
   - Keep Dawn-derived registrations connected to the Dawn root with `dependencyRoots`.
3. If the Windows WebGPU plugin pipeline changes the downloaded DXC release, update the DirectXShaderCompiler release
   registration to match `tools/ci_build/github/azure-pipelines/stages/plugin-win-webgpu-stage.yml`.
4. Run:

   ```powershell
   python cgmanifests/webgpu/validate_webgpu_cgmanifest.py
   python cgmanifests/webgpu/update_webgpu_cgmanifest.py --check
   ```

The validator checks the local Dawn pin, DXC release metadata, and Dawn dependency roots without network access.
The updater's `--check` mode also resolves the pinned Dawn revision and compares every registered Dawn dependency
against its DEPS, returning a nonzero exit code for stale hashes or paths without writing. Neither replaces the
dependency selection and classification review in step 2. Use `--deps-file PATH` and `--manifest PATH` to update or
check alternate input files.

Non-git Dawn toolchain packages from CIPD/GCS, such as GN, Ninja, CMake, Go, Siso, reclient, and sysroots, are
intentionally not registered here unless they become redistributed or CG/legal guidance requires build input coverage.
They do not have stable public upstream source identities in the Dawn `DEPS` file and are not part of current WebGPU
package contents.
