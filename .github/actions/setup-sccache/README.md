# Shared compiler caching

Use `setup-sccache` after checkout, Python setup, and toolchain selection on macOS, Linux, or Windows.
It reuses preinstalled sccache (required on Windows), installs a pinned release when missing elsewhere,
restores a bounded local cache on every run, and enables writes only on `refs/heads/main`.
Main uses the cache even when empty to populate a baseline. Other refs use an exact or compatible-prefix
snapshot read-only, and bypass compiler interception when no snapshot was restored. An exact `cache-hit`
of `false` can still mean a successful prefix restore; selection uses `cache-matched-key`.

The required `key` must identify the workflow/build variant, target architecture, configuration, and
selected compiler/SDK (including CUDA when applicable). Include runner/container image identity when
it affects compatibility. Host OS/architecture and sccache version are included automatically.
Keep vcpkg caching separate. Callers may share an environment fingerprint between vcpkg and sccache
keys, as the macOS workflow does, while retaining each cache's dependency/build-specific identity.

```yaml
- name: Setup compiler cache
  id: sccache
  uses: ./.github/actions/setup-sccache
  with:
    key: ${{ steps.toolchain.outputs.cache-key }}
    cache-size: 10G

- name: Build
  id: build
  run: python tools/ci_build/build.py --update --build ${{ steps.sccache.outputs.build-flags }} ...

- name: Finish compiler cache
  if: ${{ always() && steps.sccache.outcome == 'success' }}
  uses: ./.github/actions/finish-sccache
  with:
    cache-primary-key: ${{ steps.sccache.outputs.cache-primary-key }}
    cache-hit: ${{ steps.sccache.outputs.cache-hit }}
    save: ${{ steps.build.outcome == 'success' }}
```

Call the finish action after compilation and before tests. It reports statistics even on build failure,
stops the local server, and uploads a new snapshot only for a successful main build without an exact
cache hit. Snapshot keys include the commit SHA, with compatible-prefix restores across commits.
Actions cache ref-scoping still applies; a cache created on a feature branch is not a main baseline.

The actions do not select a CMake generator or enable compiler launchers themselves. Pass the
`build-flags` output to the configuration phase. Keep the existing generator and test-only phases;
`--test` must not be changed into a rebuild. The local limit is 10G; GitHub's default snapshot
retention/eviction applies without custom deletion policies.

## Windows

Call setup after selecting the Visual C++ environment and adding the selected CUDA SDK to PATH.
The image must provide sccache 0.19.0 or newer. Setup reports its actual executable/version and hashes
the selected `cl.exe`, SDK/toolset environment, image version when available, and NVCC identity.
The workflow/job variant (including matrix axes) remains part of the key.

With Visual Studio, `build.py --use_cache --cache_tool sccache` creates build-local compiler-named
copies of sccache and sets `CLToolPath`/`CLToolExe` and `CudaToolkitNvccPath` in `CMAKE_VS_GLOBALS`.
The copies are **not** added to PATH: sccache resolves the real compiler through the selected
environment. A mismatched target architecture, explicit MSVC toolset or CUDA home is an error.
`UseMultiToolTask` keeps MSVC invocations single-source. Ninja continues using CMake launchers.
This preserves Visual Studio project generation, configuration-specific artifact directories,
native dependencies of C#/NuGet/Java/Node packages, and existing security flags.

Cache-enabled Windows builds use embedded `/Z7` compiler debug information in every configuration;
linker-generated PDBs remain available for packaging and auditing. CUDA MSBuild otherwise overrides
that with `/Zi`, so `sccache_cuda.targets` selects embedded symbols after CUDA metadata initialization.
It also spells `--machine=64` (or the selected target width) without changing its value: sccache
misinterprets the separated `64` in the SDK's default `--machine 64` as a second input.

MSVC PCH creation/use remains enabled and passes through sccache uncached (`/Yc`/`/Fp` in statistics),
including WebGPU's adapter-first include requirement. NVCC decomposition also has non-cacheable
assembly steps; judge CUDA hits by the reported CUDA/PTX/CUBIN stages, not just outer invocations.
C#/F#/VB compilation, linking, signing, packaging and JavaScript transpilation are not sccache
compiler-cache operations.

### Measuring benefit before merging

Manually dispatch `windows_x64_release_build_x64_release.yml` on the feature branch with
`benchmark_sccache=true`. After the normal build and tests, it measures three clean `onnxruntime`
target builds on the same runner: uncached, an empty read/write cache, and that cache reopened
read-only. Each phase cleans compiled outputs first, so a warm result cannot be an incremental no-op.
The benchmark uses a separate temporary local cache, never restores/publishes snapshots, and does
not change the production main-only publication or cold-PR bypass policy.

The job summary and `windows-sccache-benchmark` artifact contain timings, structured cache statistics
and configure/build logs. Configuration is timed separately. Report cache hits/misses and PCH
bypasses alongside the timings; this measures runtime compilation/linking, not setup, packaging,
tests, or snapshot transfer. It is one sequential trial per phase, not a whole-job performance claim.
The script changes the selected build tree; use it only in disposable CI/local build directories.

### Adoption inventory

| Workflow | Cached native compilation |
| --- | --- |
| `windows_x64_release_build_x64_release.yml` | x64 RelWithDebInfo, C#/NuGet/Java/Node/wheel native dependencies |
| `windows_x64_debug_build_x64_debug.yml` | x64 Debug, C#/NuGet/Java/Node/wheel native dependencies |
| `windows_x64_release_xnnpack.yml` | XNNPACK and NuGet native dependencies |
| `windows_x64_release_ep_generic_interface_build_x64_release_ep_generic_interface.yml` | Generic-interface build |
| `windows_x64_release_vitisai_build_x64_release.yml` | VitisAI build |
| `windows_build_x64_asan.yml` | Debug ASan build; existing test shards remain test-only |
| `windows_arm64_python.yml` | Native ARM64 wheel build |
| `windows_cuda.yml`, `windows_cuda_plugin.yml`, `windows_cuda_no_cudnn.yml` | CUDA and CUDA plugin builds, including targeted provider-test build |
| `windows_tensorrt.yml` | TensorRT/CUDA build |
| `windows_dml.yml`, `windows_openvino.yml` | DirectML and OpenVINO builds |
| `windows_gpu_doc_gen.yml` | CPU/CUDA/DirectML runtime and WebGPU plugin compilation |
| `windows_webgpu.yml` | Built-in vcpkg/non-vcpkg matrix, shared/static plugin, external Dawn, package-backed built-in/plugin, minimal build |

These are 21 compile jobs (22 matrix-expanded builds) across 15 workflows. Existing separate GPU test jobs and artifact
handoffs are unchanged. DirectML retains its current combined runner topology.

`windows-web-ci-workflow.yml` consumes Linux-built WASM artifacts and only transpiles/packages/tests
JavaScript. `wgsl_template_tests.yml` runs Python tests. `publish-csharp-apidocs.yml` builds managed
documentation, not native ORT. None has a cacheable native compiler path to enable. Compiler fixtures
inside `tooling_tests.yml` intentionally remain uncached: they validate build/export behavior rather
than build product binaries. Azure release/publishing/signing pipelines are outside this GitHub
Actions rollout. There is no current Windows WinML/ARM64EC/cross-compile product lane in these
workflows; the build helper requires the matching developer environment rather than silently using
the host compiler when such a lane is added.
