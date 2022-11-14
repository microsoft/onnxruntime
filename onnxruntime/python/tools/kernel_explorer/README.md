# Kernel Explorer

Kernel Explorer hooks up GPU kernel code with a Python frontend to help develop, test, profile, and auto-tune GPU kernels. The initial scope is for BERT-like models with ROCM EP.

## Build

```bash
#!/bin/bash

set -ex

build_dir="build"
config="Release"

rocm_home="/opt/rocm"

./build.sh --update \
    --build_dir ${build_dir} \
    --config ${config} \
    --cmake_extra_defines \
        CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
        onnxruntime_BUILD_KERNEL_EXPLORER=ON \
    --skip_submodule_sync --skip_tests \
    --use_rocm --rocm_home=${rocm_home} --nccl_home=${rocm_home} \
    --build_wheel

cmake --build ${build_dir}/${config} --target kernel_explorer --parallel
```

## Run

Taking `vector_add_test.py` and build configuration with `build_dir="build"` and `config="Release"` in the previous section as an example.

Set up the native library search path with the following environment variable:
```bash
export KERNEL_EXPLORER_BUILD_DIR=`realpath build/Release`
```

To test a kernel implementation, `pip install pytest` and then

```bash
pytest onnxruntime/python/tools/kernel_explorer/kernels/vector_add_test.py
```

To run the microbenchmarks:

```bash
python onnxruntime/python/tools/kernel_explorer/kernels/vector_add_test.py
```

Currently, kernel explorer mainly targets kernel developers, not the onnxruntime package end users, so it is not installed via `setup.py`.
