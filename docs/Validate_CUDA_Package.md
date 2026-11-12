# Validate CUDA kernel execution in a Python package

A Python wheel can import successfully and list `CUDAExecutionProvider` while
lacking executable CUDA kernels for the installed GPU. CPU architecture and GPU
compute capability are separate requirements: an aarch64 wheel also needs CUDA
code compatible with its target GPU.

`onnxruntime/test/python/onnxruntime_test_python_cuda_kernel.py` is a standalone
package smoke test. It generates an FP32 `Mul` model and synthetic inputs in
memory, so no downloaded models, datasets, credentials, or application code are
needed. It checks two input sizes, exact results, and profiling events proving
that `Mul` ran on the requested execution provider. Graph optimizations and CPU
fallback are disabled for the CUDA test. Missing CUDA support is a failure,
not a skipped test.

## Run against an installed wheel

Use a machine with a supported NVIDIA GPU, driver, and the CUDA/cuDNN libraries
required by the wheel. See the [CUDA execution provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
Create an isolated environment and install the wheel under test:

```bash
python3 -m venv /tmp/ort-cuda-check
. /tmp/ort-cuda-check/bin/activate
ORT_WHEEL=/absolute/path/to/the/candidate.whl
python -m pip install numpy onnx "$ORT_WHEEL"
```

Set `ORT_WHEEL` to the actual wheel filename. Do not install the CPU and GPU
distributions together: they both provide the `onnxruntime` Python module.
Copy the single test file into a directory outside the source checkout and run it
there to ensure Python imports the installed package:

```bash
cp onnxruntime/test/python/onnxruntime_test_python_cuda_kernel.py /tmp/ort-cuda-check/
cd /tmp/ort-cuda-check
python onnxruntime_test_python_cuda_kernel.py -v
```

Both `test_mul_cpu` and `test_mul_cuda` must pass. A CPU-only sanity check is
available, but does not establish GPU support:

```bash
python onnxruntime_test_python_cuda_kernel.py TestCudaKernel.test_mul_cpu -v
```

To select a GPU on a multi-GPU machine, set `CUDA_VISIBLE_DEVICES` before starting
Python. The test uses the default device visible to ONNX Runtime.

## Architecture coverage of the packages

The Linux CUDA 13 packages contain SASS only, no PTX. A GPU whose compute capability is
not covered by any entry of the architecture list fails with
`cudaErrorNoKernelImageForDevice` at its first kernel; the run fails instead of
falling back to the CPU.

The Linux aarch64 wheel (`cmake_linux_aarch64_cuda_archs` in
`tools/ci_build/github/azure-pipelines/py-cuda13-packaging-pipeline.yml`, kept in
sync with the CUDA plugin pipelines and `build_linux_python_package.sh`) is built
for `87-real;89-real;90-real;100f-real;110-real;120f-real`:

| Entry | GPUs (compute capability) |
| --- | --- |
| `87` | Jetson Orin (8.7) |
| `89` | Ada GPUs such as L4, L40S and RTX 6000 Ada (8.9) |
| `90` | GH200 (9.0) |
| `100f` | B200/GB200 (10.0) and B300/GB300 (10.3) |
| `110` | Jetson Thor (11.0) |
| `120f` | RTX 50 and RTX PRO Blackwell (12.0), DGX Spark/GB10 (12.1) |

A family-specific target (`f` suffix) runs on every GPU with the same major compute
capability and an equal or higher minor version, so one `100f` cubin replaces the
former `100a` + `103a` pair, and one `120f` cubin replaces `120a` + `121a`. Building
family-specific targets requires CMake 3.31.8 or later (4.0.2 or later in the 4.0
series); older versions reject the `f` suffix. Each architecture costs 34-80 MB of
the wheel (`sm_90a` is the largest), which must stay under the PyPI size limit, so
do not add an entry that is already covered. Ampere (8.0, 8.6) GPUs are not covered
by the aarch64 wheel. `tools/ci_build/test_cuda_package_architectures.py` checks
that the lists are in sync and cover these GPUs without redundant entries.

When the architecture list has SM80 SASS, FlashAttention is compiled for SM80 but not
for SM86 and SM89 (`onnxruntime_FLASH_ATTENTION_REUSE_SM80`, ON by default): SM86 and
SM89 GPUs run the SM80 cubin, and the kernels choose their tile configuration at run
time. This saves about 15 MB per skipped architecture, about 30 MB in each x64
package. Build with `-Donnxruntime_FLASH_ATTENTION_REUSE_SM80=OFF` to compile it for
SM86 and SM89 as well. When changing FlashAttention, also run the CUDA tests on an
SM86 and an SM89 GPU.

## Reproduce an architecture coverage failure

For example, run the CUDA test on an SM100 GPU with a Linux aarch64 CUDA 13 wheel
that omits SM100 kernels. The expected failure is a CUDA kernel launch error such
as `cudaErrorNoKernelImageForDevice` during `Mul`. Repeat the same test on the same
GPU with a wheel built with SM100 coverage. The corrected package must pass both
the result and provider checks.

Use a separate virtual environment for each wheel. Record the wheel SHA-256,
ONNX Runtime version/build information, Python version, CPU architecture, GPU
model/compute capability, driver, CUDA/cuDNN versions, and test output. Useful
commands include:

```bash
sha256sum "$ORT_WHEEL"
uname -m
getconf PAGESIZE
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
python -c 'import platform, onnxruntime as ort; print(platform.python_version()); print(ort.__version__); print(ort.get_build_info()); print(ort.__file__); print(ort.get_available_providers())'
```

An unavailable execution provider or missing CUDA library is an environment or
package failure, not evidence of the particular missing-kernel regression.
Likewise, CPU success alone is not a substitute for the CUDA test.

## Source builds and coverage limits

For a focused native aarch64 SM100 build, first follow the
[CUDA build prerequisites](https://onnxruntime.ai/docs/build/eps.html#cuda) and
set `CUDA_HOME` and `CUDNN_HOME` to compatible CUDA 13 and cuDNN installations.
Then run from the source checkout:

```bash
python tools/ci_build/build.py --config Release --build_dir build/sm100 \
  --update --build --parallel 8 --build_wheel --use_cuda \
  --cuda_home "$CUDA_HOME" --cudnn_home "$CUDNN_HOME" \
  --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=100-real'
```

Install the resulting wheel from `build/sm100/Release/dist/` into the isolated
environment above. This focused build requests SM100; validating the full
packaging architecture list still requires the corresponding packaging build.

The existing Python test runner invokes this module when `--use_cuda` and Python
tests are enabled. CMake copies it with the other Python test files. A packaging
job that only builds wheels does not execute it; validate the produced wheel on
the target GPU separately using the commands above.

This test checks basic compiled CUDA kernel execution. It does not qualify every
operator, every supported architecture, plugin execution providers, or
application-level numerical accuracy. Run it on each GPU architecture relevant
to a package change, alongside the existing provider and application tests.
