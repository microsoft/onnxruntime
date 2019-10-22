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
    if [ $BUILD_DEVICE = "nnapi" ]; then
        cmake -DCMAKE_TOOLCHAIN_FILE=/android-ndk/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc -Donnxruntime_USE_NNAPI=ON ../cmake
    else
        cmake -DCMAKE_TOOLCHAIN_FILE=/android-ndk/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc ../cmake
    fi
    make -j$(nproc)
else
    COMMON_BUILD_ARGS="--skip_submodule_sync --enable_onnx_tests --parallel --build_shared_lib --use_openmp --cmake_path /usr/bin/cmake --ctest_path /usr/bin/ctest"
    if [ $BUILD_OS = "manylinux2010" ]; then
        # FindPython3 does not work on manylinux2010 image, define things manually
        # ask python where to find includes
        COMMON_BUILD_ARGS="${COMMON_BUILD_ARGS} --cmake_extra_defines PYTHON_INCLUDE_DIR=$(python3 -c 'import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())')"
        # Python does not provide a shared library on manylinux, use another library
        COMMON_BUILD_ARGS="${COMMON_BUILD_ARGS} PYTHON_LIBRARY=/usr/lib64/librt.so"

    fi
    if [ $BUILD_DEVICE = "gpu" ]; then
        if [ $BUILD_OS = "manylinux2010" ]; then
            python3 $SCRIPT_DIR/../../build.py --build_dir /build \
                --config Debug Release $COMMON_BUILD_ARGS \
                --use_cuda \
                --cuda_home /usr/local/cuda \
                --cudnn_home /usr/local/cuda $BUILD_EXTR_PAR
        else
            _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
            python3 $SCRIPT_DIR/../../build.py --build_dir /build \
                --config Debug Release $COMMON_BUILD_ARGS \
                --use_cuda \
                --cuda_home /usr/local/cuda \
                --cudnn_home /usr/local/cudnn-$_CUDNN_VERSION/cuda $BUILD_EXTR_PAR
        fi
    elif [ $BUILD_DEVICE = "tensorrt" ]; then
        _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Release $COMMON_BUILD_ARGS \
            --use_tensorrt --tensorrt_home /workspace/tensorrt \
            --cuda_home /usr/local/cuda \
            --cudnn_home /usr/local/cuda $BUILD_EXTR_PAR
    else #cpu, ngraph and openvino
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Debug Release $COMMON_BUILD_ARGS $BUILD_EXTR_PAR
    fi
fi

