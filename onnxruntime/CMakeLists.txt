add_subdirectory(core)

if (onnxruntime_ENABLE_PYTHON)
  add_subdirectory(python)
endif()