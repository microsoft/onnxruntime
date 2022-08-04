if (onnxruntime_USE_TVM)
  message(STATUS "onnxruntime_USE_TVM: Fetch tvm for TVM EP")

  FetchContent_Declare(
    tvm
    GIT_REPOSITORY https://github.com/Deelvin/tvm.git
    GIT_TAG        ca65ad8c90cb560a4b8fd9a3699ff8cb33d2ba6a
  )

  FetchContent_GetProperties(tvm)
  if(NOT tvm_POPULATED)
    FetchContent_Populate(tvm)
    if (WIN32)
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${tvm_BINARY_DIR}/${CMAKE_BUILD_TYPE} ${tvm_SOURCE_DIR}/build
      )
    else()
      file(CREATE_LINK ${tvm_BINARY_DIR} ${tvm_SOURCE_DIR}/build SYMBOLIC)
    endif()
  endif()

  set(tvm_INCLUDE_DIRS ${tvm_SOURCE_DIR}/include)

endif()

if (onnxruntime_USE_NUPHAR)
  message(STATUS "onnxruntime_USE_NUPHAR: Fetch onnxruntime-tvm for NUPHAR EP")

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
