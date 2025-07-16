#!/bin/bash
set -ex
#Every cuda container has this $CUDA_VERSION env var set.
SHORT_CUDA_VERSION=$(echo $CUDA_VERSION | sed   's/\([[:digit:]]\+\.[[:digit:]]\+\)\.[[:digit:]]\+/\1/')

#TODO: add --update --build
BUILD_ARGS=('--config' 'Release'
              '--skip_submodule_sync'
              '--build_shared_lib'
              '--parallel' '--use_vcpkg' '--use_vcpkg_ms_internal_asset_cache' '--use_binskim_compliant_compile_flags'
              '--build_wheel'
              '--enable_onnx_tests'
              '--use_cuda'
              "--cuda_version=$SHORT_CUDA_VERSION"
              "--cuda_home=/usr/local/cuda-$SHORT_CUDA_VERSION"
              "--cudnn_home=/usr/local/cuda-$SHORT_CUDA_VERSION"
              "--use_tensorrt" "--tensorrt_home" "/usr"
              "--enable_pybind"
              "--build_java"
              "--cmake_extra_defines"
              "CMAKE_CUDA_ARCHITECTURES=75"
              "onnxruntime_BUILD_UNIT_TESTS=ON"
              "onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS=ON")

# Parse external args
for arg in "$@"; do
  case $arg in
    --cuda_minimal=ON)
      # Replace onnxruntime_BUILD_UNIT_TESTS=ON with OFF
      BUILD_ARGS=("${BUILD_ARGS[@]/onnxruntime_BUILD_UNIT_TESTS=ON/onnxruntime_BUILD_UNIT_TESTS=OFF}")
      BUILD_ARGS+=("--enable_cuda_minimal_build")
      BUILD_ARGS+=("--skip_tests")
      ;;
  esac
done

if [ -x "$(command -v ninja)" ]; then
    BUILD_ARGS+=('--cmake_generator' 'Ninja')
fi
    
if [ -d /build ]; then
    BUILD_ARGS+=('--build_dir' '/build')
else
    BUILD_ARGS+=('--build_dir' 'build')
fi

if [ -x "$(command -v ccache)" ]; then
    ccache -s;
    BUILD_ARGS+=("--use_cache")
fi
if [ -f /opt/python/cp312-cp312/bin/python3 ]; then
    PATH=/opt/python/cp312-cp312/bin:$PATH /opt/python/cp312-cp312/bin/python3 tools/ci_build/build.py "${BUILD_ARGS[@]}"
else
    python3 tools/ci_build/build.py "${BUILD_ARGS[@]}"
fi
if [ -x "$(command -v ccache)" ]; then                
    ccache -sv 
    ccache -z
fi
