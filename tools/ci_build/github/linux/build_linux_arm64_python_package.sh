#!/bin/bash
set -e -x
mkdir -p /build/dist
CFLAGS="-Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong -O3 -pipe -Wl,--strip-all"
CXXFLAGS="-Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS -fstack-protector-strong -O3 -pipe -Wl,--strip-all"


BUILD_DEVICE="CPU"
BUILD_CONFIG="Release"
PYTHON_EXES=("/opt/python/cp38-cp38/bin/python3.8" "/opt/python/cp39-cp39/bin/python3.9" "/opt/python/cp310-cp310/bin/python3.10" "/opt/python/cp311-cp311/bin/python3.11")
while getopts "d:p:" parameter_Option
do case "${parameter_Option}"
in
#GPU or CPU.
d) BUILD_DEVICE=${OPTARG};;
p) PYTHON_EXES=(${OPTARG});;
esac
done

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


BUILD_ARGS=("--build_dir" "/build" "--config" "$BUILD_CONFIG" "--update" "--build" "--skip_submodule_sync" "--parallel" "--enable_lto" "--build_wheel")

if [ "$ARCH" == "x86_64" ]; then
    #ARM build machines do not have the test data yet.
    BUILD_ARGS+=("--enable_onnx_tests")
fi

if [ "$BUILD_DEVICE" == "GPU" ]; then
    #Enable CUDA and TRT EPs.
    ONNXRUNTIME_CUDA_VERSION="11.8"
    BUILD_ARGS+=("--use_cuda" "--use_tensorrt" "--cuda_version=$ONNXRUNTIME_CUDA_VERSION" "--tensorrt_home=/usr" "--cuda_home=/usr/local/cuda-$ONNXRUNTIME_CUDA_VERSION" "--cudnn_home=/usr/local/cuda-$ONNXRUNTIME_CUDA_VERSION" "--cmake_extra_defines" "CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80")
elif [ "$BUILD_DEVICE" == "AZURE" ]; then
    BUILD_ARGS+=("--use_azure")
    if [ -f /etc/lsb-release ]; then
        # for ubuntu
        apt-get install -y libipc-system-simple-perl python3 libssl-dev
    else
        export PATH=/opt/python/cp38-cp38/bin:$PATH
    fi
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
