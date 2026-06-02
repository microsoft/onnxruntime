#!/bin/bash
set -e -o -x

while getopts a:c: parameter_Option
do case "${parameter_Option}"
in
a) ARTIFACT_DIR=${OPTARG};;
c) CUDA_MAJOR=${OPTARG};;
*) echo "Unknown option"; exit 1;;
esac
done

if [ -z "$CUDA_MAJOR" ]; then
  echo "Error: CUDA major version (-c) is required"
  exit 1
fi

uname -a

cd "$ARTIFACT_DIR"

mkdir -p "$ARTIFACT_DIR"/onnxruntime-linux-x64-tensorrt
tar zxvf "$ARTIFACT_DIR"/onnxruntime-linux-x64-tensorrt-*.tgz -C onnxruntime-linux-x64-tensorrt
rm "$ARTIFACT_DIR"/onnxruntime-linux-x64-tensorrt-*.tgz

# Rename cuda directory to gpu_cuda{MAJOR} directory
GPU_DIR_NAME="onnxruntime-linux-x64-gpu_cuda${CUDA_MAJOR}"
mkdir -p "$ARTIFACT_DIR"/"$GPU_DIR_NAME"
tar zxvf "$ARTIFACT_DIR"/onnxruntime-linux-x64-cuda-*.tgz -C "$GPU_DIR_NAME"
VERSION=$(find "$ARTIFACT_DIR"/"$GPU_DIR_NAME" -maxdepth 1 -mindepth 1 -printf '%f\n' | sed 's/onnxruntime-linux-x64-cuda-//')
mv "$ARTIFACT_DIR"/"$GPU_DIR_NAME"/* "$ARTIFACT_DIR"/"$GPU_DIR_NAME"/"${GPU_DIR_NAME}-${VERSION}"
rm "$ARTIFACT_DIR"/onnxruntime-linux-x64-cuda-*.tgz

cp onnxruntime-linux-x64-tensorrt/*/lib/libonnxruntime.so* "$GPU_DIR_NAME"/*/lib
cp onnxruntime-linux-x64-tensorrt/*/lib/libonnxruntime_providers_tensorrt.so "$GPU_DIR_NAME"/*/lib
cp onnxruntime-linux-x64-tensorrt/*/lib/libonnxruntime_providers_shared.so "$GPU_DIR_NAME"/*/lib
