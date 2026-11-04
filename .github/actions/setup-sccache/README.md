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

## Per-OS archived-storage budgets

`sccache-maintenance.yml` currently enforces a separate **150 GB decimal (150,000,000,000 bytes)**
budget per OS for GitHub's archived sccache snapshots. Each OS's architectures and build variants
using these shared actions share that OS's budget, separate from the unchanged per-run `10G`
local cache. It runs hourly, independently of any particular build workflow, so new callers
participate automatically without adding workflow names. GitHub may delay scheduled runs.
Unused space in one OS's budget does not increase another's; total managed storage can exceed
150 GB when multiple OSes participate. Logs report each OS's current and planned storage.

The budget includes well-formed `sccache-v1-<OS>-<arch>-<identity>-<commit>` snapshots on
`refs/heads/main`. OS and architecture identifiers are not restricted to the current runners.
Other refs/namespaces, vcpkg, CodeQL, and legacy `macos-sccache-v1-`/`v2-` pilot caches are
untouched and are not counted in this budget. New key formats require updating the scope matcher.

Cleanup inventories all pages in the shared namespace, then removes older main generations
oldest-first, preferentially retaining the newest snapshot per compatibility
prefix. If an OS's baselines alone exceed its budget, it warns and removes its oldest baselines
too (including a single oversized snapshot), rather than silently exceeding the limit.
Exact immutable cache IDs are deleted, and API errors fail the maintenance job.

This is a **soft cap**, not an atomic GitHub storage quota: cleanup jobs serialize with each other,
not with cache uploads. Each OS's storage can exceed 150 GB between maintenance runs or during concurrent
publication. Cleanup rechecks every OS afterward and fails visibly if any is still over budget; the next
run retries. Cache eviction may cause a cold build but does not change build/test coverage.

Only the separate trusted-main cleanup job has `actions: write`. PRs run regression tests with
read-only permissions; cleanup never checks out PR code or consumes triggering-run artifacts.
Manual dispatch defaults to dry run. For a read-only local preview with an authenticated `gh`:

```bash
GITHUB_REPOSITORY=microsoft/onnxruntime python .github/scripts/prune_sccache.py
python -m unittest discover -s .github/scripts -p test_prune_sccache.py -v
```

Omitting `--apply` never deletes caches. No live cleanup is needed to run the regression tests.
The cleanup script and tests use only Python's standard library and the preinstalled `gh` CLI.
