#!/bin/bash
set -e -o -x

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
    python3 $SCRIPT_DIR/../../build.py --build_dir /home/onnxruntimedev \
        --config Debug Release \
        --skip_submodule_sync \
        --parallel --build_shared_lib \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/local/cudnn-$_CUDNN_VERSION/cuda --build_shared_lib $BUILD_EXTR_PAR
else
    python3 $SCRIPT_DIR/../../build.py --build_dir /home/onnxruntimedev \
        --config Debug Release --build_shared_lib \
        --skip_submodule_sync  \
        --enable_pybind \
        --parallel --build_shared_lib --enable_msinternal $BUILD_EXTR_PAR
fi
rm -rf /home/onnxruntimedev/models
