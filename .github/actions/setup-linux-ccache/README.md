# Linux ccache snapshots

Use this action after building the Docker image (or installing the host build tools).
The `key` input must distinguish build variants and relevant flags/dependencies. The
action additionally scopes snapshots to the actual local Docker image ID, or the
host C/C++ compiler and ccache versions, and runner architecture. There is no
cross-toolchain fallback.

Archives contain only workspace-relative `.cache/ccache` entries. The v2 namespace
does not restore the historical home-relative archives. Run ID and attempt form a
unique publication suffix, so a compatible restore does not prevent a new snapshot
from being saved. Callers restore on all branches but explicitly save only after a
successful `main` build; PRs never publish GitHub cache archives. Normal GitHub cache
eviction applies.

Host builds use the exported `CCACHE_DIR` and `CCACHE_UMASK`. The pinned external
Docker build action mounts the checkout at `/onnxruntime_src` but explicitly sets
ccache to its home-cache mount. Pass this action's `container-prefix` output through
its `python_path_prefix` input to override that default before any ccache commands.
Preserve any existing Python setup after that prefix with `&&`. Direct Docker calls
must pass `CCACHE_DIR=/onnxruntime_src/.cache/ccache` and `CCACHE_UMASK=000`.

The cache directory is writable by both runner and container UIDs; permissive
ccache file modes are limited to compiler-cache contents, not the checkout or other
home caches. No compiler selection, PCH setting, or build flag is changed.

Focused regression and Linux cold/warm archive smoke:

```bash
python3 -m unittest tools.ci_build.test_linux_ccache -v
```

To also exercise container mounts and differing UIDs, set `ORT_CCACHE_DOCKER_IMAGE`
to a local image containing bash, gcc and ccache before running the same command.

The smoke uses a small C fixture, not an ORT build or performance measurement.

`linux_ccache_validation.yml` additionally runs real Debug extended-minimal CPU
builds when these cache workflows/helpers change in a PR. A cold job transfers only
a workspace-relative compiler-cache archive to a fresh warm runner, not build
outputs. The warm job restores at a different host directory depth and requires at
least 90% hits across at least 100 compilations with an identical Docker image.
Timings and raw statistics are uploaded as artifacts and included in job summaries.
This tests archive transfer with run-scoped artifacts; it does not publish PR caches
or claim to validate GitHub's main-branch cache visibility policy.
