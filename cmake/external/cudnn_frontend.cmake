
onnxruntime_fetchcontent_declare(
  cudnn_frontend
  URL ${DEP_URL_cudnn_frontend}
  URL_HASH SHA1=${DEP_SHA1_cudnn_frontend}
  EXCLUDE_FROM_ALL
)

set(CUDNN_FRONTEND_SKIP_JSON_LIB OFF CACHE BOOL "" FORCE)
set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(CUDNN_FRONTEND_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(CUDNN_PATH ${onnxruntime_CUDNN_HOME})

onnxruntime_fetchcontent_makeavailable(cudnn_frontend)

# Treat the cudnn_frontend headers as system headers so that warnings originating from them (e.g.
# unused static helper functions introduced in newer releases) are not promoted to errors by
# ONNX Runtime's -Werror flags. This applies to every consumer that links the cudnn_frontend target.
if(TARGET cudnn_frontend)
  get_target_property(_cudnn_frontend_include_dirs cudnn_frontend INTERFACE_INCLUDE_DIRECTORIES)
  if(_cudnn_frontend_include_dirs)
    set_target_properties(cudnn_frontend PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_cudnn_frontend_include_dirs}")
  endif()
endif()
