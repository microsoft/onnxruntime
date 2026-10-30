# Refreshing ORT build images

The manylinux CPU and CUDA images extend base images produced by the
`onnxruntime_build_scripts` repository in the aiinfra Azure DevOps organization.
Changing this directory does not rebuild or publish those upstream bases.

`tools/ci_build/get_docker_image.py` always requests a fresh base-image resolution,
including when it uses a registry-backed buildx layer cache. Ordinary builds
continue to reuse cached layers. For an authorized security refresh, include
`--no-cache` in `--docker-build-args` to rerun package installation even when the
base digest has not changed. The `DockerBuildArgs` pipeline-template parameter
forwards this option to the helper.

Update the producer and consumer Node/npm versions together: `install_os_deps.sh`
and `manylinux/install_shared_deps.sh` install their own copies and can otherwise
undo a producer update. Node archives are verified before extraction. npm is
updated as a complete distribution, not by replacing individual nested packages.

Publish and scan new upstream images before changing `BASEIMAGE` defaults or
pipeline/workflow overrides. Use verified immutable digests, preserve the
consuming image's OS/manylinux ABI and CUDA compatibility, and keep exported
CUDA-plugin metadata consistent. Rebuild and scan the derived image too; moving
a registry tag does not replace a running container.

The dependency refresh alone is not a clean-scan claim. npm 11.21.1 still bundles
`ip-address` 10.7.2, below the report's 10.7.4 target. CUDA's Nsight `nic_sampler`
also needs a patched NVIDIA payload, not a newer host Go compiler. Keep profiling
enabled and gate CUDA promotion on that vendor fix. The historical AlmaLinux9
ARM64 family must not be replaced by an AlmaLinux8 image from current producer
main.

Run the helper's source-only regression tests from the ORT repository root:

```shell
python -m unittest discover -s tools/ci_build -p test_get_docker_image.py
```
