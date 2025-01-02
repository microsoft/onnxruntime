
FetchContent_Declare(
  cudnn_frontend
  URL ${DEP_URL_cudnn_frontend}
  URL_HASH SHA1=${DEP_SHA1_cudnn_frontend}
)

set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(CUDNN_FRONTEND_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(CUDNN_PATH ${onnxruntime_CUDNN_HOME})
onnxruntime_fetchcontent_makeavailable(cudnn_frontend)
