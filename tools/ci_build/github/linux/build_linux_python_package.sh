#!/bin/bash
set -e -x

# This script invokes build.py

mkdir -p /build/dist

EXTRA_ARG=""

# Put 3.8 at the last because Ubuntu 20.04 use python 3.8 and we will upload the intermediate build files of this 
# config to Azure DevOps Artifacts and download them to a Ubuntu 20.04 machine to run the tests.
PYTHON_EXES=("/opt/python/cp39-cp39/bin/python3.9" "/opt/python/cp310-cp310/bin/python3.10" "/opt/python/cp311-cp311/bin/python3.11" "/opt/python/cp38-cp38/bin/python3.8")
while getopts "d:p:x:c:u:" parameter_Option
do case "${parameter_Option}"
in
#GPU or CPU.
d) BUILD_DEVICE=${OPTARG};;
p) PYTHON_EXES=("${OPTARG}");;
x) EXTRA_ARG=("${OPTARG:-""}");;
c) BUILD_CONFIG=${OPTARG};;
u) CUDA_VERSION=${OPTARG:-""};;
*) echo "Usage: $0 -d <GPU|CPU> [-p <python_exe_path>] [-x <extra_build_arg>] [-c <build_config>] [-u <cuda_version>]"
   exit 1;;
esac
done

BUILD_ARGS=("--build_dir" "/build" "--config" "$BUILD_CONFIG" "--update" "--build" "--skip_submodule_sync" "--parallel" "--build_wheel")

if [ "$BUILD_CONFIG" == "Debug" ]; then
    CFLAGS="-ggdb3"
    CXXFLAGS="-ggdb3"
else
    CFLAGS="-Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong -O3 -pipe -Wl,--strip-all"
    CXXFLAGS="-Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong -O3 -pipe -Wl,--strip-all"
    BUILD_ARGS+=("--enable_lto")
fi

# Depending on how the compiler has been configured when it was built, sometimes "gcc -dumpversion" shows the full version.
GCC_VERSION=$(gcc -dumpversion | cut -d . -f 1)
#-fstack-clash-protection prevents attacks based on an overlapping heap and stack.
if [ "$GCC_VERSION" -ge 8 ]; then
    CFLAGS="$CFLAGS -fstack-clash-protection"
    CXXFLAGS="$CXXFLAGS -fstack-clash-protection"
fi

ARCH=$(uname -m)

if [ "$ARCH" == "x86_64" ] && [ "$GCC_VERSION" -ge 9 ]; then
    CFLAGS="$CFLAGS -fcf-protection"
    CXXFLAGS="$CXXFLAGS -fcf-protection"
fi

echo "EXTRA_ARG:"
echo "$EXTRA_ARG"

if [ "$EXTRA_ARG" != "" ]; then
    BUILD_ARGS+=("$EXTRA_ARG")
fi

if [ "$ARCH" == "x86_64" ]; then
    #ARM build machines do not have the test data yet.
    BUILD_ARGS+=("--enable_onnx_tests")
fi

if [ "$BUILD_DEVICE" == "GPU" ] && [ "$CUDA_VERSION" != "" ]; then
    #Enable CUDA and TRT EPs.
    BUILD_ARGS+=("--nvcc_threads=1" "--use_cuda" "--use_tensorrt" "--cuda_version=$CUDA_VERSION" "--tensorrt_home=/usr" "--cuda_home=/usr/local/cuda-$CUDA_VERSION" "--cudnn_home=/usr/local/cuda-$CUDA_VERSION" "--cmake_extra_defines" "CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80")
fi

export CFLAGS
export CXXFLAGS
for PYTHON_EXE in "${PYTHON_EXES[@]}"
do
  rm -rf /build/$BUILD_CONFIG
  ${PYTHON_EXE} /onnxruntime_src/tools/ci_build/build.py "${BUILD_ARGS[@]}"

  cp /build/$BUILD_CONFIG/dist/*.whl /build/dist
done

which ccache && ccache -sv && ccache -z
