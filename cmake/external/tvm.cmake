if (onnxruntime_USE_STVM)
  message(STATUS "onnxruntime_USE_STVM: Fetch tvm for STVM.")

  FetchContent_Declare(
    tvm
    GIT_REPOSITORY https://github.com/apache/tvm.git
    GIT_TAG        v0.8.0
  )

  FetchContent_GetProperties(tvm)
  if(NOT tvm_POPULATED)
    FetchContent_Populate(tvm)
  endif()

  set(tvm_INCLUDE_DIRS ${tvm_SOURCE_DIR}/include)
  set(onnxruntime_STVM_HOME ${tvm_SOURCE_DIR})
  message(STATUS "Define onnxruntime_STVM_HOME.")
  message(STATUS ${onnxruntime_STVM_HOME})

endif()

if (onnxruntime_USE_NUPHAR)
  message(STATUS "onnxruntime_USE_NUPHAR: Fetch onnxruntime-tvm for NUPHAR.")

  FetchContent_Declare(
    tvm
    GIT_REPOSITORY https://github.com/microsoft/onnxruntime-tvm.git
    GIT_TAG        9ec2b92d180dff8877e402018b97baa574031b8b
  )

  FetchContent_GetProperties(tvm)
  if(NOT tvm_POPULATED)
    FetchContent_Populate(tvm)
  endif()

  set(tvm_INCLUDE_DIRS ${tvm_SOURCE_DIR}/include)

endif()