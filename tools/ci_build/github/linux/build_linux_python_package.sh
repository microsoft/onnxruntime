#!/bin/bash
set -e -x

# This script invokes build.py

mkdir -p /build/dist

EXTRA_ARG=""
ENABLE_CACHE=false
# Put 3.12 at the last because Ubuntu 24.04 use python 3.12 and we will upload the intermediate build files of this
# config to Azure DevOps Artifacts and download them to a Ubuntu 24.04 machine to run the tests.
PYTHON_EXES=(
  "/opt/python/cp311-cp311/bin/python3.11"
  "/opt/python/cp313-cp313/bin/python3.13"
  "/opt/python/cp313-cp313t/bin/python3.13"
  "/opt/python/cp314-cp314/bin/python3.14"
  "/opt/python/cp314-cp314t/bin/python3.14"
  "/opt/python/cp312-cp312/bin/python3.12"
  )

while getopts "d:p:x:c:" parameter_Option
do case "${parameter_Option}"
in
#GPU|WEBGPU|CPU|NPU.
d) BUILD_DEVICE=${OPTARG};;
p)
  # Check if OPTARG is empty or starts with a hyphen, indicating a missing or invalid argument for -p
  if [[ -z "${OPTARG}" || "${OPTARG}" == -* ]]; then
    echo "ERROR: Option -p requires a valid argument, not another option."
    exit 1
  else
    PYTHON_EXES=("${OPTARG}") # Use the provided argument for -p
  fi
  ;;
x) EXTRA_ARG=${OPTARG};;
c) BUILD_CONFIG=${OPTARG};;
*) echo "Usage: $0 -d <GPU|WEBGPU|CPU|NPU> [-p <python_exe_path>] [-x <extra_build_arg>] [-c <build_config>]"
   exit 1;;
esac
done

BUILD_ARGS=("--build_dir" "/build" "--config" "$BUILD_CONFIG" "--update" "--build" "--skip_submodule_sync" "--parallel" "--use_binskim_compliant_compile_flags" "--build_wheel" "--use_vcpkg" "--use_vcpkg_ms_internal_asset_cache")

if [ "$BUILD_CONFIG" != "Debug" ]; then
    BUILD_ARGS+=("--enable_lto")
fi

if command -v ccache &> /dev/null; then
    ccache --zero-stats
    BUILD_ARGS+=("--use_cache")
fi

ARCH=$(uname -m)




echo "EXTRA_ARG:"
echo "$EXTRA_ARG"

if [ "$EXTRA_ARG" != "" ]; then
    # SC2206: This is intentionally unquoted to allow multiple arguments.
    # shellcheck disable=SC2206
    BUILD_ARGS+=($EXTRA_ARG)
fi

if [ "$ARCH" == "x86_64" ]; then
    #ARM build machines do not have the test data yet.
    BUILD_ARGS+=("--enable_onnx_tests")
fi

if [ "$BUILD_DEVICE" == "GPU" ]; then
    if [ "$CUDA_VERSION" == "12.8" ]; then
        CUDA_ARCHS="60-real;70-real;75-real;80-real;86-real;90a-real;90-virtual"
    elif [ "$CUDA_VERSION" == "13.0" ]; then
        CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"
    else
        echo "Error: Unrecognized CUDA_VERSION: $CUDA_VERSION"
        exit 1
    fi

    SHORT_CUDA_VERSION=$(echo "$CUDA_VERSION" | sed   's/\([[:digit:]]\+\.[[:digit:]]\+\)\.[[:digit:]]\+/\1/')
    CUDA_HOME=/usr/local/cuda-$SHORT_CUDA_VERSION
    if [ ! -d "$CUDA_HOME" ] && [ -d /usr/local/cuda ]; then
        # Allow the cu13 packaging flow to run on images that expose a newer CUDA minor version via /usr/local/cuda.
        CUDA_HOME=/usr/local/cuda
    fi
    #Enable CUDA EP.
    BUILD_ARGS+=("--use_cuda" "--cuda_version=$SHORT_CUDA_VERSION" "--cuda_home=$CUDA_HOME" "--cudnn_home=$CUDA_HOME" "--nvcc_threads=1" "--cmake_extra_defines" "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" "onnxruntime_USE_FPA_INTB_GEMM=OFF")
    # Enable TRT EP only if TensorRT is installed.
    if [ -f /usr/include/NvInfer.h ]; then
        BUILD_ARGS+=("--use_tensorrt" "--tensorrt_home=/usr")
    elif [ "$ARCH" != "aarch64" ] && [ -f /opt/tensorrt/include/NvInfer.h ]; then
        # The aarch64 TensorRT tarball is not compatible with the packaging image's glibc baseline.
        BUILD_ARGS+=("--use_tensorrt" "--tensorrt_home=/opt/tensorrt")
    fi
fi
if [ "$BUILD_DEVICE" == "WEBGPU" ]; then
    BUILD_ARGS+=("--use_webgpu")
fi

if [ "$BUILD_DEVICE" == "NPU" ]; then
    #Enable QNN EP
    BUILD_ARGS+=("--build_shared_lib" "--use_qnn" "--qnn_home=/qnn_sdk")
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=ON -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  # Check if the Python executable or its directory exists
  if [ ! -f "$PYTHON_EXE" ]; then
    echo "WARNING: Python executable not found at $PYTHON_EXE. Skipping this version."
    continue
  fi

  # Recompile the entire onnxruntime from scratch for every single Python version.
  # TODO: It might be possible to reuse some intermediate files between different Python versions to speed up the build.
  rm -rf /build/"$BUILD_CONFIG"

  # that's a workaround for the issue that there's no python3 in the docker image
  # like xnnpack's cmakefile, it uses pythone3 to run a external command
  python3_dir=$(dirname "$PYTHON_EXE")
  ls "$python3_dir"
  ${PYTHON_EXE} -m pip install -r /onnxruntime_src/tools/ci_build/github/linux/python/requirements.txt
  PATH=$python3_dir:$PATH ${PYTHON_EXE} /onnxruntime_src/tools/ci_build/build.py "${BUILD_ARGS[@]}"
  cp /build/"$BUILD_CONFIG"/dist/*.whl /build/dist
done

if command -v ccache &> /dev/null; then
  # FIXME: can't use `-vv` for extra details b/c we're shipping with a decrepit version of ccache (3.something) that doesn't support it.
  ccache --show-stats # -vv
fi
