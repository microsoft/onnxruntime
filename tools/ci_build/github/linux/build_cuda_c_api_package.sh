#!/bin/bash
export CFLAGS="-Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong -fstack-clash-protection -fcf-protection -O3 -Wl,--strip-all"
export CXXFLAGS="-Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong -fstack-clash-protection -fcf-protection -O3 -Wl,--strip-all"
docker run --gpus all -e CFLAGS -e CXXFLAGS  -e NVIDIA_VISIBLE_DEVICES=all --rm --volume \
$BUILD_SOURCESDIRECTORY:/onnxruntime_src --volume $BUILD_BINARIESDIRECTORY:/build \
--volume /data/models:/build/models:ro --volume /data/onnx:/data/onnx:ro -e NIGHTLY_BUILD onnxruntimecuda11centosbuild \
python3 /onnxruntime_src/tools/ci_build/build.py --build_java --build_dir /build --config Release \
--skip_submodule_sync  --parallel --build_shared_lib --use_cuda --cuda_version=$CUDA_VERSION \
--cuda_home=/usr/local/cuda-$CUDA_VERSION --cudnn_home=/usr/local/cuda-$CUDA_VERSION \
--cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80'
