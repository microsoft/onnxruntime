#!/bin/bash
set -e -x

# This script invokes build.py

mkdir -p /build/dist

EXTRA_ARG=""
ENABLE_CACHE=false
# Put 3.10 at the last because Ubuntu 22.04 use python 3.10 and we will upload the intermediate build files of this
# config to Azure DevOps Artifacts and download them to a Ubuntu 22.04 machine to run the tests.
PYTHON_EXES=(
  "/opt/python/cp311-cp311/bin/python3.11"
  "/opt/python/cp312-cp312/bin/python3.12"
  "/opt/python/cp313-cp313/bin/python3.13"
  "/opt/python/cp313-cp313t/bin/python3.13t"
  "/opt/python/cp310-cp310/bin/python3.10"
  )
while getopts "d:p:x:c:e" parameter_Option
do case "${parameter_Option}"
in
#GPU|CPU|NPU.
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
e) ENABLE_CACHE=true;;
*) echo "Usage: $0 -d <GPU|CPU|NPU> [-p <python_exe_path>] [-x <extra_build_arg>] [-c <build_config>]"
   exit 1;;
esac
done



BUILD_ARGS=("--build_dir" "/build" "--config" "$BUILD_CONFIG" "--update" "--build" "--skip_submodule_sync" "--parallel" "--use_binskim_compliant_compile_flags" "--build_wheel")

if [ "$BUILD_CONFIG" != "Debug" ]; then
    BUILD_ARGS+=("--enable_lto")
fi
if [ "$ENABLE_CACHE" = true ] ; then
    BUILD_ARGS+=("--use_cache")
    ccache -s;
fi

ARCH=$(uname -m)




echo "EXTRA_ARG:"
echo "$EXTRA_ARG"

if [ "$EXTRA_ARG" != "" ]; then
    BUILD_ARGS+=("$EXTRA_ARG")
fi

if [ "$ARCH" == "x86_64" ]; then
    #ARM build machines do not have the test data yet.
    BUILD_ARGS+=("--enable_onnx_tests")
fi

if [ "$BUILD_DEVICE" == "GPU" ]; then
    SHORT_CUDA_VERSION=$(echo $CUDA_VERSION | sed   's/\([[:digit:]]\+\.[[:digit:]]\+\)\.[[:digit:]]\+/\1/')
    #Enable CUDA and TRT EPs.
    BUILD_ARGS+=("--use_cuda" "--use_tensorrt" "--cuda_version=$SHORT_CUDA_VERSION" "--tensorrt_home=/usr" "--cuda_home=/usr/local/cuda-$SHORT_CUDA_VERSION" "--cudnn_home=/usr/local/cuda-$SHORT_CUDA_VERSION" "--cmake_extra_defines" "CMAKE_CUDA_ARCHITECTURES=75;80;90")
fi

if [ "$BUILD_DEVICE" == "NPU" ]; then
    #Enable QNN EP
    BUILD_ARGS+=("--use_qnn" "--qnn_home=/qnn_sdk")
fi

export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=ON -DONNX_WERROR=OFF"

for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  rm -rf /build/"$BUILD_CONFIG"
  # that's a workaround for the issue that there's no python3 in the docker image
  # like xnnpack's cmakefile, it uses pythone3 to run a external command
  python3_dir=$(dirname "$PYTHON_EXE")
  ${PYTHON_EXE} -m pip install -r /onnxruntime_src/tools/ci_build/github/linux/python/requirements.txt
  PATH=$python3_dir:$PATH ${PYTHON_EXE} /onnxruntime_src/tools/ci_build/build.py "${BUILD_ARGS[@]}"
  cp /build/"$BUILD_CONFIG"/dist/*.whl /build/dist
done

if [ "$ENABLE_CACHE" = true ] ; then
  which ccache && ccache -sv && ccache -z
fi
