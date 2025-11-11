#!/bin/bash
set -e -x

if [ "$CUDA_VERSION" == "12.8" ]; then
    CUDA_ARCHS="60-real;70-real;75-real;80-real;90a-real;90-virtual"
elif [ "$CUDA_VERSION" == "13.0" ]; then
    CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
else
    echo "Error: Unrecognized CUDA_VERSION: $CUDA_VERSION"
    exit 1
fi

mkdir -p $HOME/.onnx
docker run -e SYSTEM_COLLECTIONURI --rm --volume /data/onnx:/data/onnx:ro --volume $BUILD_SOURCESDIRECTORY:/onnxruntime_src \
--volume $BUILD_BINARIESDIRECTORY:/build --volume /data/models:/build/models:ro \
--volume $HOME/.onnx:/home/onnxruntimedev/.onnx -e NIGHTLY_BUILD onnxruntimecuda${CUDA_VERSION_MAJOR}xtrt86build \
/bin/bash -c "/usr/bin/python3 /onnxruntime_src/tools/ci_build/build.py --build_dir /build --config Release \
--skip_tests --skip_submodule_sync --parallel --use_binskim_compliant_compile_flags --build_shared_lib --build_nodejs \
--use_webgpu --use_tensorrt --cuda_version=$CUDA_VERSION --cuda_home=/usr/local/cuda-$CUDA_VERSION \
--cudnn_home=/usr --tensorrt_home=/usr \
--cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}' --use_vcpkg --use_vcpkg_ms_internal_asset_cache \
&& cd /build/Release && make install DESTDIR=/build/installed"
