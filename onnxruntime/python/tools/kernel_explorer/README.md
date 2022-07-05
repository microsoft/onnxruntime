# Kernel Explorer

Kernel Explorer hooks up GPU kernel code with a Python frontend to help develop, test, profile, and auto-tune GPU kernels.

Kernel explorer mainly target for attention based models and the purpose is specific to it at the moment.

## Build

```bash
#!/bin/bash

set -ex

build_dir="build"
config="Release"

rocm_home="/opt/rocm"
rocm_version="5.1.1"

./build.sh --update \
    --build_dir ${build_dir} \
    --config ${config} \
    --cmake_generator Ninja \
    --cmake_extra_defines \
        CMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        CMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        CMAKE_EXPORT_COMPILE_COMMANDS=ON \
        onnxruntime_BUILD_KERNEL_EXPLORER=ON \
        onnxruntime_DISABLE_CONTRIB_OPS=ON \
        onnxruntime_DISABLE_ML_OPS=ON \
        onnxruntime_DEV_MODE=OFF \
    --skip_submodule_sync --skip_tests \
    --use_rocm --rocm_version=${rocm_version} --rocm_home=${rocm_home} --nccl_home=${rocm_home} \
    --build_wheel \

cmake --build ${build_dir}/${config} --target kernel_explorer
```
