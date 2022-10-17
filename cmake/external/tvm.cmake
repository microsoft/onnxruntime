if (onnxruntime_USE_TVM)
  message(STATUS "onnxruntime_USE_TVM: Fetch tvm for TVM EP")

  FetchContent_Declare(
    tvm
    GIT_REPOSITORY https://github.com/apache/tvm.git
    GIT_TAG        3425ed846308a456f98404c79f6df1693bed6377
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