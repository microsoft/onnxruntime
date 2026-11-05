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

The manylinux CPU, CUDA and WebGPU images inherit Node/npm from their producer
bases instead of reinstalling them. Custom `BASEIMAGE` overrides must also provide
Node/npm; the retained AlmaLinux9 ARM64 bases keep their existing versions.
`install_os_deps.sh` separately installs Node 24.22.2 and npm 11.21.1 for its
consumers, matching the refreshed producer. Node archives are verified before
extraction. npm is updated as a complete distribution, not by replacing individual
nested packages.

Build and promote upstream test images before changing `BASEIMAGE` defaults or
pipeline/workflow overrides for validation. Use verified immutable digests, preserve the
consuming image's OS/manylinux ABI and CUDA compatibility, and keep exported
CUDA-plugin metadata consistent. Rebuild and scan the derived image too; moving
a registry tag does not replace a running container. Promotion is not release
approval: complete the validation and scan gates before merging consumer changes.

## Current official base images

The Azure Pipelines, GitHub Actions overrides and Dockerfile defaults use the
compatible images promoted by producer
[official run 1525578 (20261104.1)](https://dev.azure.com/aiinfra/ONNX%20Runtime/_build/results?buildId=1525578),
source commit `59b1944ad05ee396e2dc20acf8b22e40f3d85ebb`.
The run completed successfully, and its promotion step verified all 12 families
under the cache tag `20261104.1`, replacing the earlier PR-test image pins.
Consumers pin the following immutable digests in
`onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/<family>@<digest>`,
rather than relying on a mutable tag:

| Family | Digest |
| --- | --- |
| `cpu_x64_almalinux8_gcc14` | `sha256:f227af058cce5f442b342c3826bf6ff525452349978d66055729ab140d13fc34` |
| `cpu_aarch64_almalinux8_gcc14` | `sha256:d88e018c996e9eb2df0070a6968c0ebcc03f85d817a7982af85299e2723e6808` |
| `cuda12_x64_almalinux8_gcc14` | `sha256:25bd9e1f17d68a747d593d6e32a8e9e76b3fe49ccafea0d7eb966d5b4cdc7983` |
| `cuda13_x64_almalinux8_gcc14` | `sha256:f64c252762d76a4e65ae0c5f9103031167ba05895f27cc9196dcde2bde808a9d` |
| `cpu_x64_almalinux8_gcc14_dotnet` | `sha256:316d6bb2eb3be18e816ea8a7adb1b9f931b9b36eafb8750549a825cb99e0bff8` |
| `cpu_aarch64_almalinux8_gcc14_dotnet` | `sha256:2d698f8a4154a3e0a0ec79252141d37d62ec2dd18676df01bb5d03eddf9a052b` |
| `cuda12_x64_almalinux8_gcc14_dotnet` | `sha256:515f1ca7dfbeae6e91f794c0a8b9163d2aa606143d8b6f62b5d4d00fd0c59eb9` |
| `cuda13_x64_almalinux8_gcc14_dotnet` | `sha256:2d1d88d6610917cc7dc394c7bfac4e6c9f86ce58e33d8775b5de5395c5f6606a` |

CUDA13 ARM64 overrides retain their existing AlmaLinux9 `20260323.1` base;
the new AlmaLinux8 ARM64 CUDA family is not an ABI-compatible replacement.
The new bases provide CPython 3.11--3.14, not 3.10, so CUDA, TensorRT and WebGPU
CI jobs that previously selected `/opt/python/cp310-cp310` now select
`/opt/python/cp311-cp311` consistently for build and test.

The dependency refresh alone is not a clean-scan claim. npm 11.21.1 still bundles
`ip-address` 10.7.2, below the report's 10.7.4 target. CUDA's Nsight `nic_sampler`
also needs a patched NVIDIA payload, not a newer host Go compiler. Keep profiling
enabled and retain that vendor fix as an outstanding remediation gate. The historical AlmaLinux9
ARM64 family must not be replaced by an AlmaLinux8 image from current producer
main.

Run the helper's source-only regression tests from the ORT repository root:

```shell
python -m unittest discover -s tools/ci_build -p test_get_docker_image.py
```
