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

  if (onnxruntime_USE_CUDA)
    set(USE_CUDA ${onnxruntime_CUDA_HOME} CACHE BOOL "Only defined for TVM")
    set(USE_MKLDNN ON CACHE BOOL "Only defined for TVM")
    set(USE_CUDNN ON CACHE BOOL "Only defined for TVM")
  endif()

  if (onnxruntime_USE_LLVM)
    set(USE_LLVM ON CACHE BOOL "Only defined for TVM")
    add_definitions(-DUSE_TVM_WITH_LLVM)
  endif()

  set(USE_OPENMP gnu CACHE STRING "Only defined for TVM")
  set(USE_MICRO ON CACHE BOOL "Only defined for TVM")
  set(USE_GTEST OFF CACHE BOOL "Only defined for TVM")

  message(STATUS "TVM BEFORE USE_LLVM=${USE_LLVM} USE_OPENMP=${USE_OPENMP} USE_MICRO=${USE_MICRO} USE_GTEST=${USE_GTEST} CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} USE_CUDA=${USE_CUDA}")
  message(STATUS "tvm_SOURCE_DIR=${tvm_SOURCE_DIR}")
  message(STATUS "tvm_BINARY_DIR=${tvm_BINARY_DIR}")
  # Fails due to https://github.com/apache/tvm/blob/main/cmake/modules/StandaloneCrt.cmake#L25
  # Use of relative paths.
  add_subdirectory(${tvm_SOURCE_DIR} ${tvm_BINARY_DIR})
  message(STATUS "TVM AFTER USE_LLVM=${USE_LLVM} USE_OPENMP=${USE_OPENMP} USE_MICRO=${USE_MICRO} USE_GTEST=${USE_GTEST} CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} USE_CUDA=${USE_CUDA}")

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