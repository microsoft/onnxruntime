# This is a customization file to quickly inject necessary change into pythgon state build

if(onnxruntime_USE_CUDA AND onnxruntime_USE_SPARSE_LT)
    if(onnxruntime_ENABLE_PYTHON)
      target_include_directories(onnxruntime_pybind11_state PRIVATE ${onnxruntime_CUSPARSELT_HOME}/include)
      target_compile_definitions(onnxruntime_pybind11_state PRIVATE -DUSE_CUSPARSELT)
    endif()
endif()