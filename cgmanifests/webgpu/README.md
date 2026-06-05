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

1. Update the Dawn registration to match the `dawn` entry in `cmake/deps.txt`.
2. Re-audit the Dawn dependency graph for the pinned Dawn commit:
   - Start from the Dawn commit in `cmake/deps.txt`; do not audit Dawn `main` or a different roll.
   - Inspect Dawn's `tools/fetch_dawn_dependencies.py` at that commit. For ORT's normal source-fetch path,
     `cmake/external/onnxruntime_external_deps.cmake` enables `DAWN_FETCH_DEPENDENCIES`, so the script's
     `required_submodules` list is the primary set of Dawn source dependencies fetched for the build.
   - Cross-reference each fetched submodule path with Dawn's `DEPS` file to get the public upstream repository URL,
     commit, and condition. Use public upstream identities in this manifest, not internal mirrors.
   - Compare that fetched set against this manifest. Add new fetched components, update changed commits or repository
     URLs, and remove entries that are no longer fetched or relevant unless CG/legal guidance requires keeping them.
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
   ```

The validator checks for stale Dawn and DXC pins, but it does not replace the manual dependency classification review
in step 2.

Non-git Dawn toolchain packages from CIPD/GCS, such as GN, Ninja, CMake, Go, Siso, reclient, and sysroots, are
intentionally not registered here unless they become redistributed or CG/legal guidance requires build input coverage.
They do not have stable public upstream source identities in the Dawn `DEPS` file and are not part of current WebGPU
package contents.
