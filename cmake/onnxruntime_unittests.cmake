# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(TEST_SRC_DIR ${ONNXRUNTIME_ROOT}/test)
set(TEST_INC_DIR ${ONNXRUNTIME_ROOT})
if (onnxruntime_USE_TVM)
  list(APPEND TEST_INC_DIR ${TVM_INCLUDES})
endif()

if (onnxruntime_USE_OPENVINO)
    list(APPEND TEST_INC_DIR ${OPENVINO_INCLUDE_DIR})
endif()

set(disabled_warnings)
set(extra_includes)
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
  set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "ONNXRuntimeTest")

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif(_UT_DEPENDS)
  if(_UT_DYN)
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} gtest gmock onnxruntime ${CMAKE_DL_LIBS} Threads::Threads)
  else()
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} gtest gmock ${onnxruntime_EXTERNAL_LIBRARIES})
  endif()
  onnxruntime_add_include_to_target(${_UT_TARGET} date_interface)
  target_include_directories(${_UT_TARGET} PRIVATE ${TEST_INC_DIR})
  if (onnxruntime_USE_CUDA)
    target_include_directories(${_UT_TARGET} PRIVATE ${CUDA_INCLUDE_DIRS} ${onnxruntime_CUDNN_HOME}/include)
  endif()
  if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS AND onnxruntime_ENABLE_PYTHON)
    target_compile_definitions(${_UT_TARGET} PRIVATE ENABLE_LANGUAGE_INTEROP_OPS)
  endif()  
  if (WIN32)
    if (onnxruntime_USE_CUDA)
      # disable a warning from the CUDA headers about unreferenced local functions
      if (MSVC)
        target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd4505>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4505>")
      endif()
    endif()
    target_compile_options(${_UT_TARGET} PRIVATE ${disabled_warnings})
  else()
    target_compile_options(${_UT_TARGET} PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=sign-compare>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
  endif()

  set(TEST_ARGS)
  if (onnxruntime_GENERATE_TEST_REPORTS)
    # generate a report file next to the test program
    list(APPEND TEST_ARGS
      "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>")
  endif(onnxruntime_GENERATE_TEST_REPORTS)

  add_test(NAME ${_UT_TARGET}
    COMMAND ${_UT_TARGET} ${TEST_ARGS}
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
    )
endfunction(AddTest)

#Do not add '${TEST_SRC_DIR}/util/include' to your include directories directly
#Use onnxruntime_add_include_to_target or target_link_libraries, so that compile definitions
#can propagate correctly.

file(GLOB onnxruntime_test_utils_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/util/include/*.h"
  "${TEST_SRC_DIR}/util/*.cc"
)

file(GLOB onnxruntime_test_common_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/common/*.cc"
  "${TEST_SRC_DIR}/common/*.h"
  "${TEST_SRC_DIR}/common/logging/*.cc"
  "${TEST_SRC_DIR}/common/logging/*.h"
  )

file(GLOB onnxruntime_test_ir_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/ir/*.cc"
  "${TEST_SRC_DIR}/ir/*.h"
  )

file(GLOB onnxruntime_test_optimizer_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/optimizer/*.cc"
  "${TEST_SRC_DIR}/optimizer/*.h"
  )

set(onnxruntime_test_framework_src_patterns
  "${TEST_SRC_DIR}/framework/*.cc"
  "${TEST_SRC_DIR}/framework/*.h"
  "${TEST_SRC_DIR}/platform/*.cc"
  )

if(WIN32)
  list(APPEND onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/windows/*.cc"
    "${TEST_SRC_DIR}/platform/windows/logging/*.cc" )
endif()

if(onnxruntime_USE_CUDA)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/framework/cuda/*)
endif()

set(onnxruntime_test_providers_src_patterns
  "${TEST_SRC_DIR}/providers/*.h"
  "${TEST_SRC_DIR}/providers/*.cc"
  "${TEST_SRC_DIR}/framework/TestAllocatorManager.cc"
  "${TEST_SRC_DIR}/framework/TestAllocatorManager.h"
  "${TEST_SRC_DIR}/framework/test_utils.cc"
  "${TEST_SRC_DIR}/framework/test_utils.h"
  )
if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
  list(APPEND onnxruntime_test_providers_src_patterns
    "${TEST_SRC_DIR}/contrib_ops/*.h"
    "${TEST_SRC_DIR}/contrib_ops/*.cc")
endif()

if(onnxruntime_USE_AUTOML)
  list(APPEND onnxruntime_test_providers_src_patterns
    "${TEST_SRC_DIR}/automl_ops/*.h"
    "${TEST_SRC_DIR}/automl_ops/*.cc")
endif()

file(GLOB onnxruntime_test_providers_src CONFIGURE_DEPENDS
  ${onnxruntime_test_providers_src_patterns})
file(GLOB_RECURSE onnxruntime_test_providers_cpu_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/providers/cpu/*"
  )
list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cpu_src})

if (onnxruntime_USE_NGRAPH)
  file(GLOB_RECURSE onnxruntime_test_providers_ngraph_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/ngraph/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_ngraph_src})
endif()

if (onnxruntime_USE_NNAPI)
  file(GLOB_RECURSE onnxruntime_test_providers_nnapi_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/nnapi/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_nnapi_src})
endif()

# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(onnxruntime_test_common_libs
  onnxruntime_test_utils
  onnxruntime_common
)

set(onnxruntime_test_ir_libs
  onnxruntime_test_utils
  onnxruntime_graph
  onnxruntime_common
)

set(onnxruntime_test_optimizer_libs
  onnxruntime_test_utils
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnxruntime_common
)

set(onnxruntime_test_framework_libs
  onnxruntime_test_utils_for_framework
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnxruntime_common
  onnxruntime_mlas
  )

set(onnxruntime_test_server_libs
  onnxruntime_test_utils_for_framework
  onnxruntime_test_utils_for_server
)

if(WIN32)
    list(APPEND onnxruntime_test_framework_libs Advapi32)
endif()

set (onnxruntime_test_providers_dependencies ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(onnxruntime_USE_CUDA)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cuda)
endif()

if(onnxruntime_USE_MKLDNN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_mkldnn)
endif()

if(onnxruntime_USE_NGRAPH)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_ngraph)
endif()

if(onnxruntime_USE_OPENVINO)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_openvino)
endif()

if(onnxruntime_USE_NNAPI)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nnapi)
endif()

if(onnxruntime_USE_AUTOML)
   list(APPEND onnxruntime_test_providers_dependencies automl_featurizers)
endif()

if(onnxruntime_USE_DML)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_dml)
endif()

file(GLOB_RECURSE onnxruntime_test_tvm_src CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/test/tvm/*.h"
  "${ONNXRUNTIME_ROOT}/test/tvm/*.cc"
  )

file(GLOB_RECURSE onnxruntime_test_openvino_src
  "${ONNXRUNTIME_ROOT}/test/openvino/*.h"
  "${ONNXRUNTIME_ROOT}/test/openvino/*.cc"
 )

if(onnxruntime_USE_NUPHAR)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/framework/nuphar/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_nuphar)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nuphar)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_nuphar)
endif()

if(onnxruntime_USE_ACL)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_acl)
endif()

if (onnxruntime_ENABLE_MICROSOFT_INTERNAL)
  include(onnxruntime_unittests_internal.cmake)
endif()

set(ONNXRUNTIME_TEST_LIBS
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MKLDNN}
    ${PROVIDERS_TENSORRT}
    ${PROVIDERS_NGRAPH}
    ${PROVIDERS_OPENVINO}
    ${PROVIDERS_NUPHAR}
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_DML}
    ${PROVIDERS_ACL}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
)

set(onnxruntime_test_providers_libs
    onnxruntime_test_utils_for_framework
    ${ONNXRUNTIME_TEST_LIBS}
  )

if(onnxruntime_USE_TENSORRT)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/tensorrt/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_tensorrt)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_tensorrt)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_tensorrt)
endif()

if(onnxruntime_USE_NNAPI)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/nnapi/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_nnapi)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nnapi)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_nnapi)
endif()

if(WIN32)
  if (onnxruntime_USE_TVM)
    list(APPEND disabled_warnings ${DISABLED_WARNINGS_FOR_TVM})
  endif()
endif()

file(GLOB onnxruntime_test_framework_src CONFIGURE_DEPENDS
  ${onnxruntime_test_framework_src_patterns}
  )

#with auto initialize onnxruntime
add_library(onnxruntime_test_utils_for_framework ${onnxruntime_test_utils_src})
onnxruntime_add_include_to_target(onnxruntime_test_utils_for_framework onnxruntime_framework gtest onnx onnx_proto)
if (onnxruntime_USE_FULL_PROTOBUF)
  target_compile_definitions(onnxruntime_test_utils_for_framework PRIVATE USE_FULL_PROTOBUF=1)
endif()
if (onnxruntime_USE_MKLDNN)
  target_compile_definitions(onnxruntime_test_utils_for_framework PUBLIC USE_MKLDNN=1)
endif()
add_dependencies(onnxruntime_test_utils_for_framework ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_test_utils_for_framework PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE ${eigen_INCLUDE_DIRS} ${ONNXRUNTIME_ROOT})
# Add the define for conditionally using the framework Environment class in TestEnvironment
target_compile_definitions(onnxruntime_test_utils_for_framework PUBLIC "HAVE_FRAMEWORK_LIB")
set_target_properties(onnxruntime_test_utils_for_framework PROPERTIES FOLDER "ONNXRuntimeTest")

#without auto initialize onnxruntime
add_library(onnxruntime_test_utils ${onnxruntime_test_utils_src})
onnxruntime_add_include_to_target(onnxruntime_test_utils onnxruntime_framework gtest onnx onnx_proto)
if (onnxruntime_USE_FULL_PROTOBUF)
  target_compile_definitions(onnxruntime_test_utils PRIVATE USE_FULL_PROTOBUF=1)
endif()
if (onnxruntime_USE_MKLDNN)
  target_compile_definitions(onnxruntime_test_utils PUBLIC USE_MKLDNN=1)
endif()
add_dependencies(onnxruntime_test_utils ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_test_utils PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE ${eigen_INCLUDE_DIRS} ${ONNXRUNTIME_ROOT})
set_target_properties(onnxruntime_test_utils PROPERTIES FOLDER "ONNXRuntimeTest")

if (SingleUnitTestProject)
  set(all_tests ${onnxruntime_test_common_src} ${onnxruntime_test_ir_src} ${onnxruntime_test_optimizer_src} ${onnxruntime_test_framework_src} ${onnxruntime_test_providers_src})
  set(all_dependencies ${onnxruntime_test_providers_dependencies} )

  if (onnxruntime_USE_TVM)
    list(APPEND all_tests ${onnxruntime_test_tvm_src})
  endif()
  if (onnxruntime_USE_OPENVINO)
    list(APPEND all_tests ${onnxruntime_test_openvino_src})
  endif()
  # we can only have one 'main', so remove them all and add back the providers test_main as it sets
  # up everything we need for all tests
  file(GLOB_RECURSE test_mains CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/*/test_main.cc"
    )
  list(REMOVE_ITEM all_tests ${test_mains})
  list(APPEND all_tests "${TEST_SRC_DIR}/providers/test_main.cc")

  # this is only added to onnxruntime_test_framework_libs above, but we use onnxruntime_test_providers_libs for the onnxruntime_test_all target.
  # for now, add it here. better is probably to have onnxruntime_test_providers_libs use the full onnxruntime_test_framework_libs
  # list given it's built on top of that library and needs all the same dependencies.
  if(WIN32)
    list(APPEND onnxruntime_test_providers_libs Advapi32)
  endif()

  AddTest(
    TARGET onnxruntime_test_all
    SOURCES ${all_tests}
    LIBS ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
    DEPENDS ${all_dependencies}
  )

  # the default logger tests conflict with the need to have an overall default logger
  # so skip in this type of
  target_compile_definitions(onnxruntime_test_all PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)

  if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
    target_link_libraries(onnxruntime_test_all PRIVATE onnxruntime_language_interop onnxruntime_pyop)
  endif()

  set(test_data_target onnxruntime_test_all)
else()
  AddTest(
    TARGET onnxruntime_test_common
    SOURCES ${onnxruntime_test_common_src}
    LIBS ${onnxruntime_test_common_libs}
    DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
  )

  AddTest(
    TARGET onnxruntime_test_ir
    SOURCES ${onnxruntime_test_ir_src}
    LIBS ${onnxruntime_test_ir_libs}
    DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
  )

  AddTest(
    TARGET onnxruntime_test_optimizer
    SOURCES ${onnxruntime_test_optimizer_src}
    LIBS ${onnxruntime_test_optimizer_libs}
    DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
  )

  AddTest(
    TARGET onnxruntime_test_framework
    SOURCES ${onnxruntime_test_framework_src}
    LIBS ${onnxruntime_test_framework_libs}
    # code smell! see if CPUExecutionProvider should move to framework so onnxruntime_providers isn't needed.
    DEPENDS ${onnxruntime_test_providers_dependencies}
  )

  AddTest(
    TARGET onnxruntime_test_providers
    SOURCES ${onnxruntime_test_providers_src}
    LIBS ${onnxruntime_test_providers_libs}
    DEPENDS ${onnxruntime_test_providers_dependencies}
  )

  set(test_data_target onnxruntime_test_ir)
endif()  # SingleUnitTestProject

# standalone test for inference session without environment
# the normal test executables set up a default runtime environment, which we don't want here
if(NOT ipo_enabled)
  #TODO: figure out why this test doesn't work with gcc LTO
AddTest(
  TARGET onnxruntime_test_framework_session_without_environment_standalone
  SOURCES "${TEST_SRC_DIR}/framework/inference_session_without_environment/inference_session_without_environment_standalone_test.cc" "${TEST_SRC_DIR}/framework/test_main.cc"
  LIBS  onnxruntime_test_utils ${ONNXRUNTIME_TEST_LIBS}
  DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
)
endif()

#
# onnxruntime_ir_graph test data
#
set(TEST_DATA_SRC ${TEST_SRC_DIR}/testdata)
set(TEST_DATA_DES $<TARGET_FILE_DIR:${test_data_target}>/testdata)

# Copy test data from source to destination.
add_custom_command(
  TARGET ${test_data_target} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${TEST_DATA_SRC}
  ${TEST_DATA_DES})
if(WIN32)
  if (onnxruntime_USE_MKLDNN)
    list(APPEND onnx_test_libs mkldnn)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_DLL_PATH} $<TARGET_FILE_DIR:${test_data_target}>
      )
  endif()
  if (onnxruntime_USE_MKLML)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${MKLML_LIB_DIR}/${MKLML_SHARED_LIB} ${MKLML_LIB_DIR}/${IOMP5MD_SHARED_LIB}
      $<TARGET_FILE_DIR:${test_data_target}>
    )
  endif()
  if (onnxruntime_USE_OPENVINO)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${OPENVINO_CPU_EXTENSION_DIR}/${OPENVINO_CPU_EXTENSION_LIB}
      $<TARGET_FILE_DIR:${test_data_target}>
    )
  endif()
  if (onnxruntime_USE_NGRAPH)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${ngraph_LIBRARIES}/
      $<TARGET_FILE_DIR:${test_data_target}>
    )
  endif()
  if (onnxruntime_USE_TVM)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tvm> $<TARGET_FILE_DIR:${test_data_target}>
      )
  endif()
endif()

add_library(onnx_test_data_proto ${TEST_SRC_DIR}/proto/tml.proto)
if(WIN32)
    target_compile_options(onnx_test_data_proto PRIVATE "/wd4125" "/wd4456")
endif()
add_dependencies(onnx_test_data_proto onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/tml.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()
onnxruntime_add_include_to_target(onnx_test_data_proto onnx_proto)
target_include_directories(onnx_test_data_proto PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx)
set_target_properties(onnx_test_data_proto PROPERTIES FOLDER "ONNXRuntimeTest")
onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS ${ONNXRUNTIME_ROOT}/core/protobuf TARGET onnx_test_data_proto)

set(onnx_test_runner_src_dir ${TEST_SRC_DIR}/onnx)
set(onnx_test_runner_common_srcs
  ${onnx_test_runner_src_dir}/TestResultStat.cc
  ${onnx_test_runner_src_dir}/TestResultStat.h
  ${onnx_test_runner_src_dir}/testenv.h
  ${onnx_test_runner_src_dir}/FixedCountFinishCallback.h
  ${onnx_test_runner_src_dir}/TestCaseResult.cc
  ${onnx_test_runner_src_dir}/TestCaseResult.h
  ${onnx_test_runner_src_dir}/testenv.cc
  ${onnx_test_runner_src_dir}/heap_buffer.h
  ${onnx_test_runner_src_dir}/heap_buffer.cc
  ${onnx_test_runner_src_dir}/OrtValueList.h
  ${onnx_test_runner_src_dir}/runner.h
  ${onnx_test_runner_src_dir}/runner.cc
  ${onnx_test_runner_src_dir}/TestCase.cc
  ${onnx_test_runner_src_dir}/TestCase.h
  ${onnx_test_runner_src_dir}/onnxruntime_event.h
  ${onnx_test_runner_src_dir}/sync_api.h
  ${onnx_test_runner_src_dir}/sync_api.cc
  ${onnx_test_runner_src_dir}/callback.h
  ${onnx_test_runner_src_dir}/callback.cc
  ${onnx_test_runner_src_dir}/mem_buffer.h
  ${onnx_test_runner_src_dir}/tensorprotoutils.h
  ${onnx_test_runner_src_dir}/tensorprotoutils.cc)

if(WIN32)
  set(wide_get_opt_src_dir ${TEST_SRC_DIR}/win_getopt/wide)
  add_library(win_getopt_wide ${wide_get_opt_src_dir}/getopt.cc ${wide_get_opt_src_dir}/include/getopt.h)
  target_include_directories(win_getopt_wide INTERFACE ${wide_get_opt_src_dir}/include)
  set_target_properties(win_getopt_wide PROPERTIES FOLDER "ONNXRuntimeTest")
  set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs})
  set(GETOPT_LIB_WIDE win_getopt_wide)
endif()

add_library(onnx_test_runner_common ${onnx_test_runner_common_srcs})
onnxruntime_add_include_to_target(onnx_test_runner_common onnxruntime_common onnxruntime_framework onnxruntime_test_utils onnx onnx_proto)
add_dependencies(onnx_test_runner_common onnx_test_data_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnx_test_runner_common PRIVATE ${eigen_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx ${ONNXRUNTIME_ROOT} ${REPO_ROOT}/cmake/external/re2)
set_target_properties(onnx_test_runner_common PROPERTIES FOLDER "ONNXRuntimeTest")

set(onnx_test_libs
  onnxruntime_test_utils
  ${ONNXRUNTIME_TEST_LIBS}
  onnx_test_data_proto
  re2)

list(APPEND onnx_test_libs ${onnxruntime_EXTERNAL_LIBRARIES} libprotobuf) # test code uses delimited parsing and hence needs to link with the full protobuf

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  list(APPEND onnx_test_libs onnxruntime_language_interop onnxruntime_pyop)
endif()

add_executable(onnx_test_runner ${onnx_test_runner_src_dir}/main.cc)
target_link_libraries(onnx_test_runner PRIVATE onnx_test_runner_common ${GETOPT_LIB_WIDE} ${onnx_test_libs})
target_include_directories(onnx_test_runner PRIVATE ${ONNXRUNTIME_ROOT})
set_target_properties(onnx_test_runner PROPERTIES FOLDER "ONNXRuntimeTest")

if (onnxruntime_USE_TVM)
  if (WIN32)
    target_link_options(onnx_test_runner PRIVATE "/STACK:4000000")
  endif()
endif()

install(TARGETS onnx_test_runner
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

if(onnxruntime_BUILD_BENCHMARKS)
  add_executable(onnxruntime_benchmark ${TEST_SRC_DIR}/onnx/microbenchmark/main.cc ${TEST_SRC_DIR}/onnx/microbenchmark/modeltest.cc)
  target_include_directories(onnxruntime_benchmark PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_graph_header} benchmark)
  if(WIN32)
    target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd4141>"
                      "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4141>")
  endif()
  target_link_libraries(onnxruntime_benchmark PRIVATE onnx_test_runner_common benchmark ${onnx_test_libs})
  add_dependencies(onnxruntime_benchmark ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_benchmark PROPERTIES FOLDER "ONNXRuntimeTest")
endif()

if(WIN32)
  target_compile_options(onnx_test_runner_common PRIVATE -D_CRT_SECURE_NO_WARNINGS)
endif()

add_test(NAME onnx_test_pytorch_converted
  COMMAND onnx_test_runner ${PROJECT_SOURCE_DIR}/external/onnx/onnx/backend/test/data/pytorch-converted)
add_test(NAME onnx_test_pytorch_operator
  COMMAND onnx_test_runner ${PROJECT_SOURCE_DIR}/external/onnx/onnx/backend/test/data/pytorch-operator)

#perf test runner
set(onnxruntime_perf_test_src_dir ${TEST_SRC_DIR}/perftest)
set(onnxruntime_perf_test_src_patterns
"${onnxruntime_perf_test_src_dir}/*.cc"
"${onnxruntime_perf_test_src_dir}/*.h")

if(WIN32)
  list(APPEND onnxruntime_perf_test_src_patterns
    "${onnxruntime_perf_test_src_dir}/windows/*.cc"
    "${onnxruntime_perf_test_src_dir}/windows/*.h" )
else ()
  list(APPEND onnxruntime_perf_test_src_patterns
    "${onnxruntime_perf_test_src_dir}/posix/*.cc"
    "${onnxruntime_perf_test_src_dir}/posix/*.h" )
endif()

file(GLOB onnxruntime_perf_test_src CONFIGURE_DEPENDS
  ${onnxruntime_perf_test_src_patterns}
  )
add_executable(onnxruntime_perf_test ${onnxruntime_perf_test_src} ${ONNXRUNTIME_ROOT}/core/framework/path_lib.cc)

target_include_directories(onnxruntime_perf_test PRIVATE ${onnx_test_runner_src_dir} ${ONNXRUNTIME_ROOT}
        ${eigen_INCLUDE_DIRS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir}
        ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx)
if (WIN32)
  target_compile_options(onnxruntime_perf_test PRIVATE ${disabled_warnings})
  SET(SYS_PATH_LIB shlwapi)
endif()

if (onnxruntime_BUILD_SHARED_LIB)
  set(onnxruntime_perf_test_libs onnxruntime_test_utils onnx_test_runner_common onnxruntime_common re2
          onnx_test_data_proto onnx_proto libprotobuf ${GETOPT_LIB_WIDE} onnxruntime ${onnxruntime_EXTERNAL_LIBRARIES}
          ${SYS_PATH_LIB} ${CMAKE_DL_LIBS})
  if(onnxruntime_USE_NSYNC)
    list(APPEND onnxruntime_perf_test_libs nsync_cpp)
  endif()
  target_link_libraries(onnxruntime_perf_test PRIVATE ${onnxruntime_perf_test_libs} Threads::Threads)
  if(tensorflow_C_PACKAGE_PATH)
    target_include_directories(onnxruntime_perf_test PRIVATE ${tensorflow_C_PACKAGE_PATH}/include)
    target_link_directories(onnxruntime_perf_test PRIVATE ${tensorflow_C_PACKAGE_PATH}/lib)
    target_link_libraries(onnxruntime_perf_test PRIVATE tensorflow)
    target_compile_definitions(onnxruntime_perf_test PRIVATE HAVE_TENSORFLOW)
  endif()
else()
  target_link_libraries(onnxruntime_perf_test PRIVATE onnx_test_runner_common ${GETOPT_LIB_WIDE} ${onnx_test_libs})
endif()
set_target_properties(onnxruntime_perf_test PROPERTIES FOLDER "ONNXRuntimeTest")

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS AND NOT onnxruntime_BUILD_SHARED_LIB)
  target_link_libraries(onnxruntime_perf_test PRIVATE onnxruntime_language_interop onnxruntime_pyop)
endif()

if (onnxruntime_USE_TVM)
  if (WIN32)
    target_link_options(onnxruntime_perf_test PRIVATE "/STACK:4000000")
  endif()
endif()

# Opaque API test can not be a part of the shared lib tests since it is using
# C++ internals apis to register custom type, kernel and schema. It also can not
# a part of providers unit tests since it requires its own environment.
set(opaque_api_test_srcs ${ONNXRUNTIME_ROOT}/test/opaque_api/test_opaque_api.cc)

AddTest(
  TARGET opaque_api_test
  SOURCES ${opaque_api_test_srcs}
  LIBS ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
  DEPENDS ${onnxruntime_test_providers_dependencies}
)

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  target_link_libraries(opaque_api_test PRIVATE onnxruntime_language_interop onnxruntime_pyop)
endif()


# shared lib
if (onnxruntime_BUILD_SHARED_LIB)
  add_library(onnxruntime_mocked_allocator ${ONNXRUNTIME_ROOT}/test/util/test_allocator.cc)
  target_include_directories(onnxruntime_mocked_allocator PUBLIC ${ONNXRUNTIME_ROOT}/test/util/include)
  set_target_properties(onnxruntime_mocked_allocator PROPERTIES FOLDER "ONNXRuntimeTest")

  #################################################################
  # test inference using shared lib
  set (ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR "${ONNXRUNTIME_ROOT}/test/shared_lib")
  set (onnxruntime_shared_lib_test_SRC
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_fixture.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_inference.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_session_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_run_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_allocator.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_nontensor_types.cc)
  if(onnxruntime_RUN_ONNX_TESTS)
    list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_io_types.cc)
  endif()
  if (NOT(${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
    #for some reason, these tests are failing. Need investigation.
    if (onnxruntime_USE_FULL_PROTOBUF)
      list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_model_loading.cc)
    endif()
  endif()
  set(onnxruntime_shared_lib_test_LIBS onnxruntime_mocked_allocator onnxruntime_test_utils onnxruntime_common
          onnx_proto)
  if(onnxruntime_USE_NSYNC)
    list(APPEND onnxruntime_shared_lib_test_LIBS nsync_cpp)
  endif()
  AddTest(DYN
          TARGET onnxruntime_shared_lib_test
          SOURCES ${onnxruntime_shared_lib_test_SRC}
          LIBS ${onnxruntime_shared_lib_test_LIBS}
          protobuf::libprotobuf
          DEPENDS ${all_dependencies}
  )
endif()

if (onnxruntime_BUILD_SERVER)
  file(GLOB onnxruntime_test_server_src
    "${TEST_SRC_DIR}/server/unit_tests/*.cc"
    "${TEST_SRC_DIR}/server/unit_tests/*.h"
  )

  file(GLOB onnxruntime_integration_test_server_src
    "${TEST_SRC_DIR}/server/integration_tests/*.py"
  )
  if(NOT WIN32)
    if(HAS_UNUSED_PARAMETER)
      set_source_files_properties("${TEST_SRC_DIR}/server/unit_tests/json_handling_tests.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
      set_source_files_properties("${TEST_SRC_DIR}/server/unit_tests/converter_tests.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
      set_source_files_properties("${TEST_SRC_DIR}/server/unit_tests/util_tests.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
      set_source_files_properties("${TEST_SRC_DIR}/server/unit_tests/prediction_service_impl_test.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
      set_source_files_properties("${TEST_SRC_DIR}/server/unit_tests/executor_test.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    endif()
  endif()
  
  add_library(onnxruntime_test_utils_for_server ${onnxruntime_test_server_src})
  onnxruntime_add_include_to_target(onnxruntime_test_utils_for_server onnxruntime_test_utils_for_framework gtest gmock onnx onnx_proto server_proto server_grpc_proto)
  add_dependencies(onnxruntime_test_utils_for_server onnxruntime_server_lib onnxruntime_server_http_core_lib Boost ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_test_utils_for_server PUBLIC ${Boost_INCLUDE_DIR} ${REPO_ROOT}/cmake/external/re2 ${CMAKE_CURRENT_BINARY_DIR}/onnx ${ONNXRUNTIME_ROOT}/server ${ONNXRUNTIME_ROOT}/server/http ${ONNXRUNTIME_ROOT}/server/http/core  ${ONNXRUNTIME_ROOT}/server/grpc ${ONNXRUNTIME_ROOT}/server ${ONNXRUNTIME_ROOT}/server/core PRIVATE ${ONNXRUNTIME_ROOT})
 if (onnxruntime_USE_OPENVINO)
   message(${OPENVINO_INCLUDE_DIR})
   target_include_directories(onnxruntime_test_utils_for_server PUBLIC ${OPENVINO_INCLUDE_DIR} ${OPENVINO_TBB_INCLUDE_DIR})
 endif()
  if(UNIX)
    target_compile_options(onnxruntime_test_utils_for_server PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=sign-compare>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
  endif()
  target_link_libraries(onnxruntime_test_utils_for_server ${Boost_LIBRARIES} spdlog::spdlog server_grpc_proto)


  AddTest(
    TARGET onnxruntime_server_tests
    SOURCES ${onnxruntime_test_server_src}
    LIBS ${onnxruntime_test_server_libs} server_proto server_grpc_proto onnxruntime_server_lib ${onnxruntime_test_providers_libs}
    DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
  )

  onnxruntime_protobuf_generate(
          APPEND_PATH IMPORT_DIRS ${REPO_ROOT}/cmake/external/protobuf/src ${ONNXRUNTIME_ROOT}/server/protobuf ${ONNXRUNTIME_ROOT}/core/protobuf
          PROTOS ${ONNXRUNTIME_ROOT}/server/protobuf/predict.proto ${ONNXRUNTIME_ROOT}/server/protobuf/onnx-ml.proto
          LANGUAGE python
          TARGET onnxruntime_server_tests
          OUT_VAR server_test_py)
          
  set(grpc_py "${CMAKE_CURRENT_BINARY_DIR}/prediction_service_pb2_grpc.py")

  add_custom_command(
    TARGET onnxruntime_server_tests
    COMMAND $<TARGET_FILE:protobuf::protoc>
    ARGS 
      --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
      --plugin=protoc-gen-grpc="${_GRPC_PY_PLUGIN_EXECUTABLE}"
      -I ${grpc_proto_path}
      "${grpc_proto}"
    DEPENDS "${grpc_proto}"
    COMMENT "Running ${_GRPC_PY_PLUGIN_EXECUTABLE} on ${grpc_proto}"
    )

  add_custom_command(
    TARGET onnxruntime_server_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/server_test
    COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_integration_test_server_src}
      ${CMAKE_CURRENT_BINARY_DIR}/server_test/
      COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_BINARY_DIR}/onnx_ml_pb2.py
      ${CMAKE_CURRENT_BINARY_DIR}/server_test/
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_BINARY_DIR}/predict_pb2.py
      ${CMAKE_CURRENT_BINARY_DIR}/server_test/
    COMMAND ${CMAKE_COMMAND} -E copy
      ${grpc_py}
      ${CMAKE_CURRENT_BINARY_DIR}/server_test/
  )

endif()

#some ETW tools
if(WIN32 AND onnxruntime_ENABLE_INSTRUMENT)
    add_executable(generate_perf_report_from_etl ${ONNXRUNTIME_ROOT}/tool/etw/main.cc ${ONNXRUNTIME_ROOT}/tool/etw/eparser.h ${ONNXRUNTIME_ROOT}/tool/etw/eparser.cc ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.h ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.cc)
    target_compile_definitions(generate_perf_report_from_etl PRIVATE "_CONSOLE" "_UNICODE" "UNICODE")
    target_link_libraries(generate_perf_report_from_etl PRIVATE tdh Advapi32)

    add_executable(compare_two_sessions ${ONNXRUNTIME_ROOT}/tool/etw/compare_two_sessions.cc ${ONNXRUNTIME_ROOT}/tool/etw/eparser.h ${ONNXRUNTIME_ROOT}/tool/etw/eparser.cc ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.h ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.cc)
    target_compile_definitions(compare_two_sessions PRIVATE "_CONSOLE" "_UNICODE" "UNICODE")
    target_link_libraries(compare_two_sessions PRIVATE ${GETOPT_LIB_WIDE} tdh Advapi32)
endif()

add_executable(onnxruntime_mlas_test ${TEST_SRC_DIR}/mlas/unittest.cpp)
target_include_directories(onnxruntime_mlas_test PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT})
set(onnxruntime_mlas_test_libs onnxruntime_mlas onnxruntime_common)
if(onnxruntime_USE_NSYNC)
  list(APPEND onnxruntime_mlas_test_libs nsync_cpp)
endif()
list(APPEND onnxruntime_mlas_test_libs Threads::Threads)
target_link_libraries(onnxruntime_mlas_test PRIVATE ${onnxruntime_mlas_test_libs})
set_target_properties(onnxruntime_mlas_test PROPERTIES FOLDER "ONNXRuntimeTest")
