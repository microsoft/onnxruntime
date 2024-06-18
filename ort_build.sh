#!/bin/bash

bash ./build.sh \
    --config Release --enable_training --build_wheel --update --build --cmake_generator Ninja --skip_tests \
    --use_rocm --rocm_version=5.4.0 --rocm_home /opt/rocm --enable_nccl \
    --nccl_home /opt/rocm --allow_running_as_root \
    --parallel --skip_tests --skip_submodule_sync \
    --enable_rocm_profiling \
    --cmake_extra_defines CMAKE_HIP_ARCHITECTURES=gfx90a

pip uninstall -y onnxruntime-training onnxruntime-gpu onnxruntime
pip install build/Linux/Release/dist/onnxruntime_training-1.19.0+rocm540.profiling-cp38-cp38-linux_x86_64.whl
python -m torch_ort.configure
