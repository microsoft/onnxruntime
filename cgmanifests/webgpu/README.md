# WebGPU Component Governance manifest

This directory contains the WebGPU-specific Component Governance manifest for ONNX Runtime. It covers Dawn and the
Dawn-derived dependency graph used when building the WebGPU Execution Provider.

The dependencies in `cgmanifest.json` are optional for ONNX Runtime as a whole. Vanilla ORT builds that do not enable
WebGPU should not treat this manifest as a global dependency list. WebGPU packaging and NOTICE-generation pipelines
should explicitly select this manifest in addition to any global ORT Component Governance metadata they already scan.

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

1. Update the Dawn registration to match the `dawn` entry in `cmake\deps.txt`.
2. Re-audit the pinned upstream Dawn `DEPS` file and update Dawn-derived registrations, comments, and
   `dependencyRoots`.
3. If the Windows WebGPU plugin pipeline changes the downloaded DXC release, update the DirectXShaderCompiler release
   registration to match `tools\ci_build\github\azure-pipelines\stages\plugin-win-webgpu-stage.yml`.
4. Run:

   ```powershell
   python tools\python\validate_webgpu_cgmanifest.py
   ```

Non-git Dawn toolchain packages from CIPD/GCS, such as GN, Ninja, CMake, Go, Siso, reclient, and sysroots, are
intentionally not registered here unless they become redistributed or CG/legal guidance requires build input coverage.
They do not have stable public upstream source identities in the Dawn `DEPS` file and are not part of current WebGPU
package contents.
