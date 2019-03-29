#!/bin/bash
set -e -o -x

id

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"

while getopts c:d:x: parameter_Option
do case "${parameter_Option}"
in
d) BUILD_DEVICE=${OPTARG};;
x) BUILD_EXTR_PAR=${OPTARG};;
esac
done

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
