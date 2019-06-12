#!/bin/bash
set -e -o -x

id

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"

while getopts d:x:o: parameter_Option
do case "${parameter_Option}"
in
d) BUILD_DEVICE=${OPTARG};;
x) BUILD_EXTR_PAR=${OPTARG};;
o) BUILD_OS=${OPTARG};;
esac
done

if [ $BUILD_OS = "android" ]; then
    pushd /onnxruntime_src
    mkdir build-android && cd build-android
    cmake -DCMAKE_TOOLCHAIN_FILE=/android-ndk/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc ../cmake
    make -j$(nproc)
else
    COMMON_BUILD_ARGS="--skip_submodule_sync --enable_onnx_tests --parallel --build_shared_lib --use_openmp --cmake_path /usr/bin/cmake --ctest_path /usr/bin/ctest"
    if [ $BUILD_DEVICE = "gpu" ]; then
        _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Debug Release $COMMON_BUILD_ARGS \
            --use_cuda \
            --cuda_home /usr/local/cuda \
            --cudnn_home /usr/local/cudnn-$_CUDNN_VERSION/cuda $BUILD_EXTR_PAR
    elif [ $BUILD_DEVICE = "tensorrt" ]; then
        _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Release $COMMON_BUILD_ARGS \
            --use_tensorrt --tensorrt_home /workspace/tensorrt \
            --cuda_home /usr/local/cuda \
            --cudnn_home /usr/local/cuda $BUILD_EXTR_PAR
    else #cpu and ngraph
        python3 $SCRIPT_DIR/../../../python/onnx_test_data_utils.py --action dump_pb --input /usr/local/lib/python3.5/dist-packages/onnx/backend/test/data/node/test_cast_STRING_to_FLOAT/test_data_set_0
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Debug Release $COMMON_BUILD_ARGS $BUILD_EXTR_PAR
    fi
fi
