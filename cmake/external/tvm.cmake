if (onnxruntime_USE_TVM)
  message(STATUS "onnxruntime_USE_TVM: Fetch tvm for TVM EP")

  FetchContent_Declare(
    tvm
    URL ${DEP_URL_tvm}
    URL_HASH SHA1=${DEP_SHA1_tvm}
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