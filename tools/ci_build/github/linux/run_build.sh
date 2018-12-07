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

if [ -z "$AZURE_BLOB_KEY" ]; then
  echo "AZURE_BLOB_KEY is blank"
  BUILD_EXTR_PAR="${BUILD_EXTR_PAR}"
  echo "Extra parameters: ${BUILD_EXTR_PAR}"
else
  echo "Downloading test data from azure"
  mkdir -p /home/onnxruntimedev/models/
  azcopy --recursive --source:https://onnxruntimetestdata.blob.core.windows.net/onnx-model-zoo-20181206 --destination:/home/onnxruntimedev/models/ --source-key:$AZURE_BLOB_KEY
  BUILD_EXTR_PAR="${BUILD_EXTR_PAR} --enable_onnx_tests"
fi

if [ $BUILD_DEVICE = "gpu" ]; then
    _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
    python3 $SCRIPT_DIR/../../build.py --build_dir /home/onnxruntimedev \
        --config Debug Release \
        --skip_submodule_sync \
        --parallel --build_shared_lib \
        --use_cuda \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/local/cudnn-$_CUDNN_VERSION/cuda --build_shared_lib $BUILD_EXTR_PAR
    /home/onnxruntimedev/Release/onnx_test_runner -e cuda /data/onnx
else
    python3 $SCRIPT_DIR/../../build.py --build_dir /home/onnxruntimedev \
        --config Debug Release --build_shared_lib \
        --skip_submodule_sync \
        --enable_pybind \
        --parallel --use_mkldnn --build_shared_lib $BUILD_EXTR_PAR
    /home/onnxruntimedev/Release/onnx_test_runner /data/onnx
fi
