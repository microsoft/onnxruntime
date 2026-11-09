# Refreshing ORT build images

The manylinux CPU and CUDA images extend separately maintained upstream base
images. Changing this directory does not rebuild or publish those upstream bases.

`tools/ci_build/get_docker_image.py` always requests a fresh base-image resolution,
including when it uses a registry-backed buildx layer cache. Ordinary builds
continue to reuse cached layers. For an authorized security refresh, include
`--no-cache` in `--docker-build-args` to rerun package installation even when the
base digest has not changed. The `DockerBuildArgs` pipeline-template parameter
forwards this option to the helper.

The manylinux CPU, CUDA and WebGPU images inherit Node/npm from their producer
bases instead of reinstalling them. Custom `BASEIMAGE` overrides must also provide
Node/npm.

Build and promote upstream test images before changing `BASEIMAGE` defaults or
pipeline/workflow overrides for validation. Use verified immutable digests, preserve the
consuming image's OS/manylinux ABI and CUDA compatibility, and keep exported
CUDA-plugin metadata consistent. Rebuild and scan the derived image too; moving
a registry tag does not replace a running container. Promotion is not release
approval: complete the validation and scan gates before merging consumer changes.

## Central base-image configuration

[`base-images.json`](base-images.json) is the source of truth for image
repositories, immutable digests, promotion tags and known producer build IDs.
Its `dockerfile_defaults` map selects a family for each managed Dockerfile.
For routine base refreshes, update only the affected `images` entries in this
manifest. Tags are provenance metadata; builds always use `repository@digest`.

ADO passes `BaseImageFamily` through `get-docker-image-steps.yml` to
`get_docker_image.py --base-image-family`. GitHub uses the local
`.github/actions/build-docker-image` action, optionally selecting a
`base-image-family` (or `base_image_family` in `reusable_linux_build.yml`).
Both call `resolve_docker_base_image.py`, using the Dockerfile's configured
default when no family or explicit `BASEIMAGE` argument is supplied.
Unknown families and conflicting family/explicit-image selections fail.

The resolver CLI writes a single line to stdout: with `--family` alone, an
immutable `repository@sha256:digest` reference; with `--dockerfile`, shell-quoted
Docker build arguments, retaining supplied arguments and adding the configured
`BASEIMAGE` if needed. The GitHub wrapper captures that line and appends
`build-args=<resolved arguments>` to `$GITHUB_OUTPUT`, exposing
`steps.resolve.outputs.build-args` to the next step. Resolution failures exit
before any step output is written.

The GitHub wrapper resolves the digest before invoking the existing Docker
action, so the digest enters its build-argument cache key. A manifest update
therefore invalidates the appropriate derived-image cache without changing the
existing registry publishing policy. CUDA plugin metadata also records the
resolved immutable reference so downstream tests use the packaging run's base,
not the current manifest's selection.

Managed Dockerfiles intentionally declare `ARG BASEIMAGE` without a pinned
default. From the repository root, use the helper for local builds:

```shell
python tools/ci_build/get_docker_image.py \
  --dockerfile tools/ci_build/github/linux/docker/Dockerfile.manylinux2_28_cuda \
  --context tools/ci_build/github/linux/docker \
  --repository ort-cuda-local \
  --base-image-family cuda13_x64_almalinux8_gcc14 \
  --docker-build-args="--build-arg TRT_VERSION="
```

Omit `--base-image-family` to use the manifest's Dockerfile default.
Without `--container-registry`, this helper does not push an image.
For a raw `docker build`, explicitly supply `--build-arg BASEIMAGE=<reference>`;
Docker itself cannot read the manifest. Explicit image arguments remain
supported for custom images and historical packaging metadata.

## Current official base images

The manifest records the nine consumed official AlmaLinux8 and AlmaLinux9 families alongside
their tags, digests and producer build IDs. Its `producer_builds` entries record
the source commits for those builds. ADO and GitHub consumers
select these families rather than duplicating their pins or current release
provenance in this document or CI files.

CUDA13 ARM64 selections use the refreshed AlmaLinux9 family from the same
official producer build as the AlmaLinux8 families. Its repository now uses the
shared promotion namespace rather than the historical image's namespace.
The AlmaLinux8 ARM64 CUDA family is not an ABI-compatible replacement.
The new bases provide CPython 3.11--3.14, not 3.10, so CUDA, TensorRT and WebGPU
CI jobs that previously selected `/opt/python/cp310-cp310` now select
`/opt/python/cp311-cp311` consistently for build and test.

The dependency refresh alone is not a clean-scan claim. npm 11.21.1 still bundles
`ip-address` 10.7.2, below the report's 10.7.4 target. CUDA's Nsight `nic_sampler`
also needs a patched NVIDIA payload, not a newer host Go compiler. Keep profiling
enabled and retain that vendor fix as an outstanding remediation gate.
