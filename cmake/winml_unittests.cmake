# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(TEST_SRC_DIR ${REPO_ROOT}/winml/test)
set(TEST_INC_DIR ${REPO_ROOT}/winml/test/common)

function(AddTest)
  cmake_parse_arguments(_UT "DYN" "TARGET" "LIBS;SOURCES;DEPENDS" ${ARGN})
  if(_UT_LIBS)
    list(REMOVE_DUPLICATES _UT_LIBS)
  endif()
  list(REMOVE_DUPLICATES _UT_SOURCES)

  if (_UT_DEPENDS)
    list(REMOVE_DUPLICATES _UT_DEPENDS)
  endif(_UT_DEPENDS)

  add_executable(${_UT_TARGET} ${_UT_SOURCES})

  source_group(TREE ${TEST_SRC_DIR} FILES ${_UT_SOURCES})
  set_target_properties(${_UT_TARGET} PROPERTIES
    FOLDER "WinMLTest"
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif(_UT_DEPENDS)
  target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} gtest gmock)
  target_include_directories(${_UT_TARGET} PRIVATE ${TEST_INC_DIR})
  target_precompiled_header(${_UT_TARGET} ${REPO_ROOT}/winml/test/common/pch.h) # FIXME

  add_test(NAME ${_UT_TARGET}
    COMMAND ${_UT_TARGET}
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
    )
endfunction(AddTest)

file(GLOB winml_test_api_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/api/*.cc"
  "${TEST_SRC_DIR}/api/*.h"
)

AddTest(
  TARGET winml_test_api
  SOURCES ${winml_test_api_src}
)

# if (SingleUnitTestProject)
#   set(all_tests ${onnxruntime_test_common_src} ${onnxruntime_test_ir_src} ${onnxruntime_test_optimizer_src} ${onnxruntime_test_framework_src} ${onnxruntime_test_providers_src})
#   set(all_dependencies ${onnxruntime_test_providers_dependencies} )

#   if (onnxruntime_USE_TVM)
#     list(APPEND all_tests ${onnxruntime_test_tvm_src})
#   endif()
#   if (onnxruntime_USE_OPENVINO)
#     list(APPEND all_tests ${onnxruntime_test_openvino_src})
#   endif()
#   # we can only have one 'main', so remove them all and add back the providers test_main as it sets
#   # up everything we need for all tests
#   file(GLOB_RECURSE test_mains CONFIGURE_DEPENDS
#     "${TEST_SRC_DIR}/*/test_main.cc"
#     )
#   list(REMOVE_ITEM all_tests ${test_mains})
#   list(APPEND all_tests "${TEST_SRC_DIR}/providers/test_main.cc")

#   # this is only added to onnxruntime_test_framework_libs above, but we use onnxruntime_test_providers_libs for the onnxruntime_test_all target.
#   # for now, add it here. better is probably to have onnxruntime_test_providers_libs use the full onnxruntime_test_framework_libs
#   # list given it's built on top of that library and needs all the same dependencies.
#   if(WIN32)
#     list(APPEND onnxruntime_test_providers_libs Advapi32)
#   endif()

#   AddTest(
#     TARGET onnxruntime_test_all
#     SOURCES ${all_tests}
#     LIBS ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
#     DEPENDS ${all_dependencies}
#   )

#   # the default logger tests conflict with the need to have an overall default logger
#   # so skip in this type of
#   target_compile_definitions(onnxruntime_test_all PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)

#   if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
#     target_link_libraries(onnxruntime_test_all PRIVATE onnxruntime_language_interop onnxruntime_pyop)
#   endif()

#   set(test_data_target onnxruntime_test_all)
# else()
#   AddTest(
#     TARGET onnxruntime_test_ir
#     SOURCES ${onnxruntime_test_ir_src}
#     LIBS ${onnxruntime_test_ir_libs}
#     DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
#   )

#   AddTest(
#     TARGET onnxruntime_test_optimizer
#     SOURCES ${onnxruntime_test_optimizer_src}
#     LIBS ${onnxruntime_test_optimizer_libs}
#     DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
#   )

#   AddTest(
#     TARGET onnxruntime_test_framework
#     SOURCES ${onnxruntime_test_framework_src}
#     LIBS ${onnxruntime_test_framework_libs}
#     # code smell! see if CPUExecutionProvider should move to framework so onnxruntime_providers isn't needed.
#     DEPENDS ${onnxruntime_test_providers_dependencies}
#   )

#   AddTest(
#     TARGET onnxruntime_test_providers
#     SOURCES ${onnxruntime_test_providers_src}
#     LIBS ${onnxruntime_test_providers_libs}
#     DEPENDS ${onnxruntime_test_providers_dependencies}
#   )

#   set(test_data_target onnxruntime_test_ir)
# endif()  # SingleUnitTestProject
