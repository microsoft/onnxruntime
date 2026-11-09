# Shared compiler caching

Use `setup-sccache` after checkout, Python setup, and toolchain selection on macOS, Linux, or Windows.
It installs a pinned sccache release, restores a bounded local cache on every run, and enables writes only
on `refs/heads/main`. Other refs compile cache misses without storing them.

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
  run: python tools/ci_build/build.py --update --build --use_cache --cache_tool sccache ...

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
`--use_cache --cache_tool sccache` in build flags. Windows launcher support in `build.py` currently
requires Ninja; validate the selected compiler/debug-symbol configuration when adopting it.
