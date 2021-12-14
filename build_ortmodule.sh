#!/bin/bash -ex

PATH=/opt/cmake/bin:$PATH

#build ort
mkdir ./ortmodule_build
build_type=Debug
rm -rf ./ortmodule_build/$build_type/*.so
rm -rf ./ortmodule_build/$build_type/dist/*
bash ./build.sh \
    --build_dir=./ortmodule_build \
    --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ \
    --cuda_version=11.1 \
    --use_cuda --config $build_type --update --build \
    --build_wheel \
    --parallel \
    --enable_training --skip_tests --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) CMAKE_CUDA_ARCHITECTURES=70 \
    --mpi_home=/usr/local/mpi \
    --enable_cuda_line_info  2>&1 | tee build_log

sudo env "PATH=$PATH" pip uninstall -y onnxruntime-training
sudo env "PATH=$PATH" pip install ./ortmodule_build/$build_type/dist/*.whl
sudo env "PATH=$PATH" pip install torch-ort
sudo env "PATH=$PATH" python /home/zhijxu/anaconda3/envs/ortmodule/lib/python3.7/site-packages/torch_ort/configure/__main__.py
