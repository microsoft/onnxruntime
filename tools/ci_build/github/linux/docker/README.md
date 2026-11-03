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

Keep producer and consumer Node/npm versions aligned when rolling out new bases:
`install_os_deps.sh` and `manylinux/install_shared_deps.sh` install Node 24.22.2
and npm 11.21.1 themselves, matching the producer draft. Node archives are
verified before extraction. npm is updated as a complete distribution, not by
replacing individual nested packages.

Build and promote upstream test images before changing `BASEIMAGE` defaults or
pipeline/workflow overrides for validation. Use verified immutable digests, preserve the
consuming image's OS/manylinux ABI and CUDA compatibility, and keep exported
CUDA-plugin metadata consistent. Rebuild and scan the derived image too; moving
a registry tag does not replace a running container. Test-image pins are not
release approval: complete the validation and scan gates before merging this
draft or publishing a new official producer version.

## Current validation images

The Azure Pipelines, GitHub Actions overrides and Dockerfile defaults use the
compatible images promoted by producer
[run 1522159 (20261103.2)](https://dev.azure.com/aiinfra/ONNX%20Runtime/_build/results?buildId=1522159),
source commit `4f265c4693972b5d971840b2bd732454fd3a2fd8`.
The promotion step verified all 12 families under the cache tag `pr-1522159`.
Consumers pin the following immutable digests in
`onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/<family>@<digest>`,
rather than relying on a mutable tag:

| Family | Digest |
| --- | --- |
| `cpu_x64_almalinux8_gcc14` | `sha256:2de06c32024c7783233c251e53c8e1247606184d2969f1a5e9b275e1002c6565` |
| `cpu_aarch64_almalinux8_gcc14` | `sha256:520921c97d37ae97e5663b36130b91a79dd2d7d570087386ef4ce12004568281` |
| `cuda12_x64_almalinux8_gcc14` | `sha256:6adedc35de9bdc383f02949bf167902abcf6447887644ea786cf37a7ff6f2e4a` |
| `cuda13_x64_almalinux8_gcc14` | `sha256:601d4e1c7de614e3b65d9fcaee33efb090f4da17c7be42bf19613b983f9573d3` |
| `cpu_x64_almalinux8_gcc14_dotnet` | `sha256:0f8e29b82f43e7092ef4ff70c5ea70fb7c3ebb376816640fe2c177b966323982` |
| `cpu_aarch64_almalinux8_gcc14_dotnet` | `sha256:5c24c8da952587b1429fc9e8f43fa365fc5f8c8090efeca444b25ae6347fa159` |
| `cuda12_x64_almalinux8_gcc14_dotnet` | `sha256:d7b6f542b0469a04661f3e28dbcb4597eb63cc2d16af1450025760afcadfda86` |
| `cuda13_x64_almalinux8_gcc14_dotnet` | `sha256:30bd89829b6906501bed65e33919f9bd7be7dbb71ef554cd12768d772a8227d3` |

CUDA13 ARM64 overrides retain their existing AlmaLinux9 `20260323.1` base;
the new AlmaLinux8 ARM64 CUDA family is not an ABI-compatible replacement.
The new bases provide CPython 3.11--3.14, not 3.10, so CUDA, TensorRT and WebGPU
CI jobs that previously selected `/opt/python/cp310-cp310` now select
`/opt/python/cp311-cp311` consistently for build and test.

The dependency refresh alone is not a clean-scan claim. npm 11.21.1 still bundles
`ip-address` 10.7.2, below the report's 10.7.4 target. CUDA's Nsight `nic_sampler`
also needs a patched NVIDIA payload, not a newer host Go compiler. Keep profiling
enabled and gate official CUDA release on that vendor fix. The historical AlmaLinux9
ARM64 family must not be replaced by an AlmaLinux8 image from current producer
main.

Run the helper's source-only regression tests from the ORT repository root:

```shell
python -m unittest discover -s tools/ci_build -p test_get_docker_image.py
```
