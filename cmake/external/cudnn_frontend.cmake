include(FetchContent)
FetchContent_Declare(
  cudnn_frontend
  URL ${DEP_URL_cudnn_frontend}
  URL_HASH SHA1=${DEP_SHA1_cudnn_frontend}
  SOURCE_DIR ${BUILD_DIR_NO_CONFIG}/_deps/cudnn_frontend-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/deps/cudnn_frontend-build
  DOWNLOAD_DIR ${BUILD_DIR_NO_CONFIG}/_deps/cudnn_frontend-download
)

set(CUDNN_FRONTEND_BUILD_SAMPLES OFF)
set(CUDNN_FRONTEND_BUILD_UNIT_TESTS OFF)
set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF)
set(CUDNN_PATH ${onnxruntime_CUDNN_HOME})
FetchContent_MakeAvailable(cudnn_frontend)
