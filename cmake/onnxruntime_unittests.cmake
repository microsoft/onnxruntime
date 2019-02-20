# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(TEST_SRC_DIR ${ONNXRUNTIME_ROOT}/test)
set(TEST_INC_DIR ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${onnxruntime_CUDNN_HOME}/include)
if (onnxruntime_USE_TVM)
  list(APPEND TEST_INC_DIR ${TVM_INCLUDES})
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
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS} eigen)
  endif(_UT_DEPENDS)
  if(_UT_DYN)
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} gtest gmock onnxruntime Threads::Threads)
  else()
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} gtest gmock debug ${onnxruntime_EXTERNAL_LIBRARIES_DEBUG} optimized ${onnxruntime_EXTERNAL_LIBRARIES})
  endif()
  onnxruntime_add_include_to_target(${_UT_TARGET} gsl eigen)
  target_include_directories(${_UT_TARGET} PRIVATE ${TEST_INC_DIR})

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

#Check whether C++17 header file <filesystem> is present
include(CheckIncludeFiles)
check_include_files("filesystem" HAS_FILESYSTEM_H LANGUAGE CXX)
check_include_files("experimental/filesystem" HAS_EXPERIMENTAL_FILESYSTEM_H LANGUAGE CXX)

#Do not add '${TEST_SRC_DIR}/util/include' to your include directories directly
#Use onnxruntime_add_include_to_target or target_link_libraries, so that compile definitions
#can propagate correctly.

file(GLOB onnxruntime_test_utils_src
  "${TEST_SRC_DIR}/util/include/*.h"
  "${TEST_SRC_DIR}/util/*.cc"
)

file(GLOB onnxruntime_test_common_src
  "${TEST_SRC_DIR}/common/*.cc"
  "${TEST_SRC_DIR}/common/*.h"
  "${TEST_SRC_DIR}/common/logging/*.cc"
  "${TEST_SRC_DIR}/common/logging/*.h"
  )

file(GLOB onnxruntime_test_ir_src
  "${TEST_SRC_DIR}/ir/*.cc"
  "${TEST_SRC_DIR}/ir/*.h"
  )

set(onnxruntime_test_framework_src_patterns
  "${TEST_SRC_DIR}/framework/*.cc"
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
  "${TEST_SRC_DIR}/contrib_ops/*.h"
  "${TEST_SRC_DIR}/contrib_ops/*.cc"
  "${TEST_SRC_DIR}/providers/*.h"
  "${TEST_SRC_DIR}/providers/*.cc"
  "${TEST_SRC_DIR}/framework/TestAllocatorManager.cc"
  "${TEST_SRC_DIR}/framework/TestAllocatorManager.h"
  )

file(GLOB onnxruntime_test_providers_src ${onnxruntime_test_providers_src_patterns})
file(GLOB_RECURSE onnxruntime_test_providers_cpu_src
  "${TEST_SRC_DIR}/providers/cpu/*"
  )
list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cpu_src})

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

set(onnxruntime_test_framework_libs
  onnxruntime_test_utils_for_framework
  onnxruntime_optimizer
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnxruntime_common
  onnxruntime_mlas
  )


if(WIN32)
    list(APPEND onnxruntime_test_framework_libs Advapi32)
elseif(HAS_FILESYSTEM_H OR HAS_EXPERIMENTAL_FILESYSTEM_H)
    list(APPEND onnxruntime_test_framework_libs stdc++fs)
endif()



set (onnxruntime_test_providers_dependencies ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(onnxruntime_USE_CUDA)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cuda)
endif()

if(onnxruntime_USE_MKLDNN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_mkldnn)
endif()

file(GLOB_RECURSE onnxruntime_test_tvm_src
  "${ONNXRUNTIME_ROOT}/test/tvm/*.h"
  "${ONNXRUNTIME_ROOT}/test/tvm/*.cc"
  )

if (onnxruntime_ENABLE_MICROSOFT_INTERNAL)
  include(onnxruntime_unittests_internal.cmake)
endif()

set(ONNXRUNTIME_TEST_LIBS
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MKLDNN}
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

if( NOT WIN32 AND (HAS_FILESYSTEM_H OR HAS_EXPERIMENTAL_FILESYSTEM_H))
  list(APPEND onnxruntime_test_providers_libs stdc++fs)
endif()

if(WIN32)
  if (onnxruntime_USE_TVM)
    list(APPEND disabled_warnings ${DISABLED_WARNINGS_FOR_TVM})
  endif()
endif()

file(GLOB onnxruntime_test_framework_src ${onnxruntime_test_framework_src_patterns})

#with auto initialize onnxruntime
add_library(onnxruntime_test_utils_for_framework ${onnxruntime_test_utils_src})
onnxruntime_add_include_to_target(onnxruntime_test_utils_for_framework onnxruntime_framework gtest gsl onnx onnx_proto)
add_dependencies(onnxruntime_test_utils_for_framework ${onnxruntime_EXTERNAL_DEPENDENCIES} eigen)
target_include_directories(onnxruntime_test_utils_for_framework PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE ${eigen_INCLUDE_DIRS} ${ONNXRUNTIME_ROOT})
# Add the define for conditionally using the framework Environment class in TestEnvironment
target_compile_definitions(onnxruntime_test_utils_for_framework PUBLIC "HAVE_FRAMEWORK_LIB")

#without auto initialize onnxruntime
add_library(onnxruntime_test_utils ${onnxruntime_test_utils_src})
onnxruntime_add_include_to_target(onnxruntime_test_utils onnxruntime_framework gtest gsl onnx onnx_proto)
add_dependencies(onnxruntime_test_utils ${onnxruntime_EXTERNAL_DEPENDENCIES} eigen)
target_include_directories(onnxruntime_test_utils PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE ${eigen_INCLUDE_DIRS} ${ONNXRUNTIME_ROOT})


if (SingleUnitTestProject)
  set(all_tests ${onnxruntime_test_common_src} ${onnxruntime_test_ir_src} ${onnxruntime_test_framework_src} ${onnxruntime_test_providers_src})
  set(all_dependencies ${onnxruntime_test_providers_dependencies} )

  if (onnxruntime_USE_TVM)
    list(APPEND all_tests ${onnxruntime_test_tvm_src})
  endif()
  # we can only have one 'main', so remove them all and add back the providers test_main as it sets
  # up everything we need for all tests
  file(GLOB_RECURSE test_mains "${TEST_SRC_DIR}/*/test_main.cc")
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
AddTest(
  TARGET onnxruntime_test_framework_session_without_environment_standalone
  SOURCES "${TEST_SRC_DIR}/framework/inference_session_without_environment/inference_session_without_environment_standalone_test.cc" "${TEST_SRC_DIR}/framework/test_main.cc"
  LIBS  onnxruntime_test_utils ${ONNXRUNTIME_TEST_LIBS}
  DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
)

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
      COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_LIB_DIR}/${MKLDNN_SHARED_LIB} $<TARGET_FILE_DIR:${test_data_target}>
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
  ${onnx_test_runner_src_dir}/runner.h
  ${onnx_test_runner_src_dir}/runner.cc
  ${onnx_test_runner_src_dir}/TestCase.cc
  ${onnx_test_runner_src_dir}/TestCase.h
  ${onnx_test_runner_src_dir}/path_lib.h
  ${onnx_test_runner_src_dir}/sync_api.h
  ${onnx_test_runner_src_dir}/sync_api.cc)

if(WIN32)
  set(wide_get_opt_src_dir ${TEST_SRC_DIR}/win_getopt/wide)
  add_library(win_getopt_wide ${wide_get_opt_src_dir}/getopt.cc ${wide_get_opt_src_dir}/include/getopt.h)
  target_include_directories(win_getopt_wide INTERFACE ${wide_get_opt_src_dir}/include)
  set_target_properties(win_getopt_wide PROPERTIES FOLDER "ONNXRuntimeTest")
  set(mb_get_opt_src_dir ${TEST_SRC_DIR}/win_getopt/mb)
  add_library(win_getopt_mb ${mb_get_opt_src_dir}/getopt.cc ${mb_get_opt_src_dir}/include/getopt.h)
  target_include_directories(win_getopt_mb INTERFACE ${mb_get_opt_src_dir}/include)
  set_target_properties(win_getopt_mb PROPERTIES FOLDER "ONNXRuntimeTest")

  set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs})

  set(GETOPT_LIB_WIDE win_getopt_wide)
  set(GETOPT_LIB win_getopt_mb)
else()
  set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs} ${onnx_test_runner_src_dir}/onnxruntime_event.h ${onnx_test_runner_src_dir}/simple_thread_pool.h)
  if(HAS_FILESYSTEM_H OR HAS_EXPERIMENTAL_FILESYSTEM_H)
    set(FS_STDLIB stdc++fs)
  endif()
endif()

add_library(onnx_test_runner_common ${onnx_test_runner_common_srcs})
onnxruntime_add_include_to_target(onnx_test_runner_common onnxruntime_common onnxruntime_framework onnxruntime_test_utils gsl onnx onnx_proto)
add_dependencies(onnx_test_runner_common eigen onnx_test_data_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnx_test_runner_common PRIVATE ${eigen_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx ${ONNXRUNTIME_ROOT})
set_target_properties(onnx_test_runner_common PROPERTIES FOLDER "ONNXRuntimeTest")


set(onnx_test_libs
  onnxruntime_test_utils
  ${ONNXRUNTIME_TEST_LIBS}
  onnx_test_data_proto)

if(WIN32)
  list(APPEND onnx_test_libs Pathcch)
endif()

list(APPEND onnx_test_libs ${FS_STDLIB} debug ${onnxruntime_EXTERNAL_LIBRARIES_DEBUG} optimized ${onnxruntime_EXTERNAL_LIBRARIES})


add_executable(onnx_test_runner ${onnx_test_runner_src_dir}/main.cc)
target_link_libraries(onnx_test_runner PRIVATE onnx_test_runner_common ${GETOPT_LIB_WIDE} ${onnx_test_libs})
onnxruntime_add_include_to_target(onnx_test_runner gsl)
target_include_directories(onnx_test_runner PRIVATE ${ONNXRUNTIME_ROOT})
set_target_properties(onnx_test_runner PROPERTIES FOLDER "ONNXRuntimeTest")

install(TARGETS onnx_test_runner
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

if(onnxruntime_BUILD_BENCHMARKS AND (HAS_FILESYSTEM_H OR HAS_EXPERIMENTAL_FILESYSTEM_H))
  add_executable(onnxruntime_benchmark ${TEST_SRC_DIR}/onnx/microbenchmark/main.cc ${TEST_SRC_DIR}/onnx/microbenchmark/modeltest.cc)
  target_include_directories(onnxruntime_benchmark PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_graph_header} benchmark)
  target_compile_options(onnxruntime_benchmark PRIVATE "/wd4141")
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

if(HAS_FILESYSTEM_H OR HAS_EXPERIMENTAL_FILESYSTEM_H)
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

  file(GLOB onnxruntime_perf_test_src ${onnxruntime_perf_test_src_patterns})
  add_executable(onnxruntime_perf_test ${onnxruntime_perf_test_src})

  target_include_directories(onnxruntime_perf_test PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx)
  if (WIN32)
    target_compile_options(onnxruntime_perf_test PRIVATE ${disabled_warnings})
  endif()
  onnxruntime_add_include_to_target(onnxruntime_perf_test gsl)
  target_link_libraries(onnxruntime_perf_test PRIVATE ${GETOPT_LIB} ${onnx_test_libs})
  set_target_properties(onnxruntime_perf_test PROPERTIES FOLDER "ONNXRuntimeTest")
endif()

# shared lib
if (onnxruntime_BUILD_SHARED_LIB)
  if (UNIX)
    # test custom op shared lib
    file(GLOB onnxruntime_custom_op_shared_lib_test_srcs "${ONNXRUNTIME_ROOT}/test/custom_op_shared_lib/test_custom_op.cc")
    add_library(onnxruntime_custom_op_shared_lib_test SHARED ${onnxruntime_custom_op_shared_lib_test_srcs})
    onnxruntime_add_include_to_target(onnxruntime_custom_op_shared_lib_test gsl)
    add_dependencies(onnxruntime_custom_op_shared_lib_test onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
    target_include_directories(onnxruntime_custom_op_shared_lib_test PUBLIC "${PROJECT_SOURCE_DIR}/include")
    target_link_libraries(onnxruntime_custom_op_shared_lib_test PRIVATE onnxruntime onnx onnx_proto  protobuf::libprotobuf)
    set_target_properties(onnxruntime_custom_op_shared_lib_test PROPERTIES FOLDER "ONNXRuntimeSharedLibTest")
  endif()
  add_library(onnxruntime_mocked_allocator ${ONNXRUNTIME_ROOT}/test/util/test_allocator.cc)
  target_include_directories(onnxruntime_mocked_allocator PUBLIC ${ONNXRUNTIME_ROOT}/test/util/include)
  #################################################################
  # test inference using shared lib + custom op
  set (ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR "${ONNXRUNTIME_ROOT}/test/shared_lib")
  set (onnxruntime_shared_lib_test_SRC
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_fixture.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_inference.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_session_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_run_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_allocator.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_inference.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_nontensor_types.cc)
  if(onnxruntime_RUN_ONNX_TESTS)
    list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_io_types.cc)
  endif()
  AddTest(DYN
          TARGET onnxruntime_shared_lib_test
          SOURCES ${onnxruntime_shared_lib_test_SRC}
          LIBS onnxruntime_mocked_allocator
          DEPENDS ${all_dependencies}
  )

  #demo
  if(PNG_FOUND)
    add_executable(fns_candy_style_transfer "${ONNXRUNTIME_ROOT}/test/shared_lib/fns_candy_style_transfer.c")
    target_include_directories(fns_candy_style_transfer PRIVATE "${TEST_SRC_DIR}/util/include" ${PNG_INCLUDE_DIRS})
    target_link_libraries(fns_candy_style_transfer PRIVATE onnxruntime ${PNG_LIBRARIES})
    set_target_properties(fns_candy_style_transfer PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()
endif()


add_executable(onnxruntime_mlas_test ${TEST_SRC_DIR}/mlas/unittest.cpp)
target_include_directories(onnxruntime_mlas_test PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc)
target_link_libraries(onnxruntime_mlas_test PRIVATE onnxruntime_mlas)
set_target_properties(onnxruntime_mlas_test PROPERTIES FOLDER "ONNXRuntimeTest")
