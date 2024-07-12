if (onnxruntime_USE_TVM)
  message(STATUS "onnxruntime_USE_TVM: Fetch tvm for TVM EP")

  onnxruntime_fetchcontent_declare(
    tvm
    GIT_REPOSITORY https://github.com/apache/tvm.git
    GIT_TAG        2379917985919ed3918dc12cad47f469f245be7a
    EXCLUDE_FROM_ALL
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
