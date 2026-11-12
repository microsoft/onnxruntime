# Shared compiler caching

Use `setup-sccache` after checkout, Python setup, and toolchain selection on macOS, Linux, or Windows.
It reuses the installed sccache executable, or installs the pinned v0.19.0 release if it is missing on the
host. The setup action never replaces an installed tool; its actual version is validated, logged, and included in the key.
It restores on every ref and enables writes only on `refs/heads/main`.

Use the `build-flags` output instead of unconditional launcher flags. Main always uses sccache to populate
the cache. Nonpublishing runs use it only after an exact **or compatible-prefix** restore; a complete miss
compiles directly, without a cache server. `cache-hit: false` alone is not a miss: a fallback snapshot is usable.

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

The actions do not select a CMake generator or enable compiler launchers themselves. Keep
the `build-flags` output in build commands. Windows launcher support in `build.py` currently
requires Ninja; validate the selected compiler/debug-symbol configuration when adopting it.

## Reusable Linux builds

Every caller of `reusable_linux_build.yml` uses sccache as its only compiler-cache choice: CPU
Debug/Release/MinSizeRel, x64/arm64, Python variants, CUDA (including plugins), TensorRT, OpenVINO,
and WebGPU. There is no per-variant opt-in or ccache restore/save path. Ninja, existing security/build
flags, artifacts, and test coverage are unchanged. Windows and workflows that do not call this reusable
workflow are outside this change. vcpkg caching remains independent.

Compilation runs in Docker, not on the VM host. `container-image` requires **preinstalled sccache in
that image**; missing/invalid tools fail rather than installing on the host. The seven repo-derived
Dockerfiles used by this workflow install v0.19.0 into `/usr/local/bin` before selecting the build user,
using the shared `tools/ci_build/github/linux/docker/scripts/install_sccache.sh`. It downloads the
x64/arm64 musl release with a pinned SHA256 and verifies the installed version. Shared base-image pins
and VM provisioning are unchanged. Both `build-docker-image` and `get_docker_image.py` stage the installer
at the root of each Docker build context; the action does so before hashing, so installer edits invalidate
derived-image caches. For a manual Docker build, copy the installer to the context's `install_sccache.sh`
first. Existing ccache binaries are retained for the external action's statistics commands.

The probe runs with the same
Python/toolchain PATH prefix as compilation, logs the executable/version, GCC/C++/CMake/Python/OS details
and resolved image ID, and hashes these with the job, architecture, configuration, EP list, and build
flags into the compatibility key. The image ID and CUDA selection flags also partition CUDA toolkit
versions and SDK contents. There is no
cross-toolchain restore prefix. The Docker probe runs before image inspection so registry-only image
cache hits are pulled first.

This integration depends on the pinned `run-build-script-in-docker` action's workspace mount at
`/onnxruntime_src` and runner-temp mount at `/onnxruntime_src/build`. Its `python_path_prefix` sources
`container.sh` after selecting Python, before update/build commands. The hook loads a fresh, uncached
runner-temp environment file, points sccache at the workspace-mounted `.sccache`, and starts/zeros the
server **inside each ephemeral container**. The exit trap reports statistics and stops the server before
Docker exits, preserving build failures and failing on cache lifecycle errors. The host finish action
requires the flush marker and reads the compilation statistics; it never runs a host sccache server.
The workflow also checks the actual generated C/C++ and, when configured, CUDA launchers (or their
absence on a cold nonpublishing run). CUDA and TensorRT builds use sccache for nvcc as well as C/C++.
Test execution keeps the existing Docker invocation and happens after archive publication.
The pinned external Docker action still invokes ccache's version/statistics commands internally;
these do not select a compiler launcher or restore/save a ccache archive. This change does not remove
that tool from the build images or modify the external action.

CPU/test PCH configuration is MSVC-only, and CUDA PCH is Windows-only. The WebGPU plugin uses PCH
on Linux as well: its GCC target deliberately clears the sccache launcher while retaining PCH and
the required adapter-header include order. sccache 0.19.0 can incorrectly reuse a GCC PCH after a
macro-only header change ([mozilla/sccache#2562](https://github.com/mozilla/sccache/issues/2562));
disabling direct mode does not fix that case. Other targets in that build still use sccache.

The Linux namespace is `sccache-linux-v1-`, followed by runner OS/architecture and the identity hash,
separate from macOS and the earlier pilot namespace.
Local size is 10G. Archived snapshots use GitHub Actions' default cache retention and eviction policy;
there is no custom maintenance workflow, archive budget, snapshot-count limit, or deletion API call.
Build jobs do not need `actions: write`. They restore on all refs, but save only successful main
compilations without an exact hit, after the server flushes and before tests.

To evaluate the rollout, collect native Linux cold/warm compilation duration, cache hit/miss and
noncacheable counts, restore/save duration and archive bytes. A cold PR deliberately bypasses sccache,
so successful PR CI alone cannot measure warm caching. A fresh main namespace first needs a successful
population build and then a compatible subsequent build. No performance gain is asserted by this rollout.

Focused lifecycle/selection checks:

```text
python -m unittest discover -s .github/actions/setup-sccache -p test_cache.py -v
```
