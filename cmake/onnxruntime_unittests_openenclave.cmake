# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(TEST_SRC_DIR ${ONNXRUNTIME_ROOT}/test)
set(TEST_INC_DIR ${ONNXRUNTIME_ROOT})
set(OPENENCLAVE_TEST_SRC_DIR ${TEST_SRC_DIR}/openenclave)

# Session test enclave (used in perftest and session-based unit tests)
set(session_enclave_src_dir ${OPENENCLAVE_TEST_SRC_DIR}/session_enclave)
set(session_enclave_edl_path ${session_enclave_src_dir}/shared/session.edl)

if (onnxruntime_OPENENCLAVE_BUILD_ENCLAVE)
    # Enclave image of session test enclave.
    add_custom_command(
      OUTPUT session_t.h session_t.c session_args.h
      DEPENDS ${session_enclave_edl_path}
      COMMAND openenclave::oeedger8r --trusted ${session_enclave_edl_path})
    add_executable(onnxruntime_session_test_enclave
        ${session_enclave_src_dir}/enclave/session_enclave.cc
        ${session_enclave_src_dir}/enclave/threading.cc
        ${TEST_SRC_DIR}/onnx/tensorprotoutils.cc
        ${TEST_SRC_DIR}/onnx/callback.cc
        ${CMAKE_CURRENT_BINARY_DIR}/session_t.c)
    target_link_libraries(onnxruntime_session_test_enclave PRIVATE openenclave-enclave onnxruntime_openenclave)
    target_include_directories(onnxruntime_session_test_enclave PRIVATE
        ${ONNXRUNTIME_ROOT}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/onnx)
    onnxruntime_add_include_to_target(onnxruntime_session_test_enclave onnx safeint_interface)
    add_dependencies(onnxruntime_session_test_enclave onnx_proto)
    set_target_properties(onnxruntime_session_test_enclave PROPERTIES FOLDER "ONNXRuntimeTest")
else()
    # Host wrapper of session test enclave.
    add_custom_command(
      OUTPUT session_u.h session_u.c session_args.h
      DEPENDS ${session_enclave_edl_path}
      COMMAND openenclave::oeedger8r --untrusted ${session_enclave_edl_path})
    add_library(onnxruntime_session_test_enclave_host STATIC
        ${session_enclave_src_dir}/host/session_enclave.cc
        ${session_enclave_src_dir}/host/threading.cc
        ${TEST_SRC_DIR}/onnx/tensorprotoutils.cc
        ${CMAKE_CURRENT_BINARY_DIR}/session_u.c)
    target_link_libraries(onnxruntime_session_test_enclave_host PUBLIC
        openenclave-host
        )
    target_include_directories(onnxruntime_session_test_enclave_host PUBLIC
        ${ONNXRUNTIME_ROOT}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/onnx)
    add_dependencies(onnxruntime_session_test_enclave_host onnx_proto)
    onnxruntime_add_include_to_target(onnxruntime_session_test_enclave_host onnxruntime_common onnx safeint_interface)
    # prevent symbol clash between libsgx's libprotobuf.so and ONNX RT's libprotobuf.a
    target_link_options(onnxruntime_session_test_enclave_host INTERFACE
        LINKER:--version-script=${OPENENCLAVE_TEST_SRC_DIR}/no_symbols.txt
    )
    set_target_properties(onnxruntime_session_test_enclave_host PROPERTIES FOLDER "ONNXRuntimeTest")

    # Unit tests
    function(AddOpenEnclaveTest)
      cmake_parse_arguments(_UT "" "TARGET;ENCLAVE" "LIBS;SOURCES;DEPENDS" ${ARGN})
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
      target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} gtest gmock ${onnxruntime_EXTERNAL_LIBRARIES})
      onnxruntime_add_include_to_target(${_UT_TARGET} date_interface flatbuffers)
      target_include_directories(${_UT_TARGET} PRIVATE ${TEST_INC_DIR})
      #target_compile_options(${_UT_TARGET} PRIVATE "-Wno-error=sign-compare")

      set(TEST_ARGS "${_UT_ENCLAVE}")
      if (onnxruntime_GENERATE_TEST_REPORTS)
        # generate a report file next to the test program
        list(APPEND TEST_ARGS
          "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>")
      endif(onnxruntime_GENERATE_TEST_REPORTS)

      add_test(NAME ${_UT_TARGET}
        COMMAND ${_UT_TARGET} ${TEST_ARGS}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
        )
    endfunction()

    file(GLOB onnxruntime_test_utils_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/util/include/*.h"
      "${TEST_SRC_DIR}/util/*.cc"
    )

    add_library(onnxruntime_test_oe_utils ${onnxruntime_test_utils_src})
    onnxruntime_add_include_to_target(onnxruntime_test_oe_utils onnxruntime_common onnxruntime_framework GTest::gtest GTest::gmock onnx onnx_proto safeint_interface)
    if (onnxruntime_USE_FULL_PROTOBUF)
      target_compile_definitions(onnxruntime_test_oe_utils PRIVATE USE_FULL_PROTOBUF=1)
    endif()
    add_dependencies(onnxruntime_test_oe_utils ${onnxruntime_EXTERNAL_DEPENDENCIES})
    target_compile_definitions(onnxruntime_test_oe_utils PUBLIC -DNSYNC_ATOMIC_CPP11)
    target_include_directories(onnxruntime_test_oe_utils PUBLIC "${TEST_SRC_DIR}/util/include" ${eigen_INCLUDE_DIRS} PRIVATE ${ONNXRUNTIME_ROOT} "${CMAKE_CURRENT_SOURCE_DIR}/external/nsync/public")
    set_target_properties(onnxruntime_test_oe_utils PROPERTIES FOLDER "ONNXRuntimeTest")

    set(ONNXRUNTIME_TEST_LIBS
      onnxruntime_test_oe_utils
      onnxruntime_session_test_enclave_host
      onnxruntime_session
      onnxruntime_optimizer
      onnxruntime_providers
      onnxruntime_util
      onnxruntime_framework
      onnxruntime_util
      onnxruntime_graph
      onnxruntime_common
      onnxruntime_mlas
      onnxruntime_flatbuffers
      libprotobuf
      )

    file(GLOB onnxruntime_test_oe_src CONFIGURE_DEPENDS
      ${OPENENCLAVE_TEST_SRC_DIR}/unit_tests/*
      )

    AddOpenEnclaveTest(
        TARGET onnxruntime_oe_test
        ENCLAVE ${onnxruntime_OPENENCLAVE_ENCLAVE_BUILD_DIR}/onnxruntime_session_test_enclave
        SOURCES ${onnxruntime_test_oe_src}
        LIBS ${ONNXRUNTIME_TEST_LIBS}
        DEPENDS ${onnxruntime_EXTERNAL_DEPENDENCIES}
      )
endif()