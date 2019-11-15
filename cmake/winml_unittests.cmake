# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(WINML_TEST_SRC_DIR ${REPO_ROOT}/winml/test)
set(WINML_TEST_INC_DIR
  ${REPO_ROOT}/winml/test/common
  ${REPO_ROOT}/winml/lib/Api.Image/inc
  ${REPO_ROOT}/winml/lib/Common/inc
  ${REPO_ROOT}/onnxruntime
  # ${REPO_ROOT}/cmake/external/onnx
  ${REPO_ROOT}/cmake/external/protobuf/src
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/winml_api
  ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated
  ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include)

function(AddWinMLTest)
  cmake_parse_arguments(_UT "DYN" "TARGET" "LIBS;SOURCES;DEPENDS" ${ARGN})
  if(_UT_LIBS)
    list(REMOVE_DUPLICATES _UT_LIBS)
  endif()
  list(REMOVE_DUPLICATES _UT_SOURCES)

  if (_UT_DEPENDS)
    list(REMOVE_DUPLICATES _UT_DEPENDS)
  endif(_UT_DEPENDS)

  add_executable(${_UT_TARGET} ${_UT_SOURCES})
  source_group(TREE ${WINML_TEST_SRC_DIR} FILES ${_UT_SOURCES})

  set_target_properties(${_UT_TARGET} PROPERTIES
    FOLDER "WinMLTest"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
  )

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif(_UT_DEPENDS)
  target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} gtest_main windowsapp winml_lib_image onnx ${onnxruntime_EXTERNAL_LIBRARIES} libprotobuf) # FIXME
  target_include_directories(${_UT_TARGET} PRIVATE ${WINML_TEST_INC_DIR})
#   target_precompiled_header(${_UT_TARGET} ${REPO_ROOT}/winml/test/common/pch.h)

  add_test(NAME ${_UT_TARGET}
    COMMAND ${_UT_TARGET}
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
  )
endfunction(AddWinMLTest)

file(GLOB winml_test_api_src CONFIGURE_DEPENDS
  "${WINML_TEST_SRC_DIR}/api/*.cpp"
  "${WINML_TEST_SRC_DIR}/api/*.h"
  "${WINML_TEST_SRC_DIR}/common/fileHelpers.cpp"  # FIXME move common to a separate project
  "${WINML_TEST_SRC_DIR}/common/protobufHelpers.cpp"  # FIXME move common to a separate project
  "${WINML_TEST_SRC_DIR}/common/SqueezeNetValidator.cpp"  # FIXME move common to a separate project
)

AddWinMLTest(
  TARGET winml_test_api
  SOURCES ${winml_test_api_src}
)
install(DIRECTORY "${WINML_TEST_SRC_DIR}/collateral/images/" "${WINML_TEST_SRC_DIR}/collateral/models/" "${WINML_TEST_SRC_DIR}/collateral/ModelSubdirectory" "${WINML_TEST_SRC_DIR}/api/models/" DESTINATION $<TARGET_FILE_DIR:winml_test_api>)
install(FILES "${WINML_TEST_SRC_DIR}/collateral/metaDataTestTable.xml" DESTINATION $<TARGET_FILE_DIR:winml_test_api>)
