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
    /opt/cmake/bin/cmake -DCMAKE_TOOLCHAIN_FILE=/android-ndk/build/cmake/android.toolchain.cmake -DANDROID_CPP_FEATURES=exceptions -DANDROID_PLATFORM=android-28 -DANDROID_ABI=arm64-v8a -DCMAKE_BUILD_TYPE=Release -Donnxruntime_CROSS_COMPILING=ON -Donnxruntime_BUILD_x86=OFF -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc ../cmake
    /opt/cmake/bin/cmake --build . -- -j$(nproc)
else
    if [ $BUILD_DEVICE = "gpu" ]; then
        _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Debug Release \
            --skip_submodule_sync --enable_onnx_tests \
            --parallel --build_shared_lib \
            --use_cuda --use_openmp \
            --cuda_home /usr/local/cuda \
            --cudnn_home /usr/local/cudnn-$_CUDNN_VERSION/cuda --build_shared_lib $BUILD_EXTR_PAR
    elif [ $BUILD_DEVICE = "tensorrt" ]; then
        _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Release \
            --enable_onnx_tests \
            --parallel --build_shared_lib \
            --use_tensorrt --tensorrt_home /workspace/tensorrt \
            --use_openmp \
            --cuda_home /usr/local/cuda \
            --cudnn_home /usr/local/cuda --build_shared_lib $BUILD_EXTR_PAR
    else
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Debug Release --build_shared_lib \
            --skip_submodule_sync --enable_onnx_tests \
            --build_wheel \
            --parallel --use_openmp --build_shared_lib $BUILD_EXTR_PAR
    fi
fi
