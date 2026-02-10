# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Telum Execution Provider Tests

set(TEST_SRC_DIR ${ONNXRUNTIME_ROOT}/test/providers/telum)
set(TEST_INC_DIR ${ONNXRUNTIME_ROOT}/test/providers/telum)

# Test source files
set(telum_test_src
  # Utility tests
  ${TEST_SRC_DIR}/utils/test_endian_utils.cc
  ${TEST_SRC_DIR}/test_end_to_end.cc

  # Kernel tests
  ${TEST_SRC_DIR}/kernels/test_matmul.cc
  ${TEST_SRC_DIR}/kernels/test_gemm.cc
  ${TEST_SRC_DIR}/kernels/test_elementwise.cc
  ${TEST_SRC_DIR}/kernels/test_activations.cc
  ${TEST_SRC_DIR}/kernels/test_softmax.cc
  ${TEST_SRC_DIR}/kernels/test_layer_norm.cc
)

# Test headers
set(telum_test_inc
  ${TEST_INC_DIR}/test_utils.h
)

# Create test executable
add_executable(onnxruntime_telum_test
  ${telum_test_src}
  ${telum_test_inc}
)

# Include directories
target_include_directories(onnxruntime_telum_test PRIVATE
  ${ONNXRUNTIME_ROOT}
  ${ONNXRUNTIME_ROOT}/../include/onnxruntime
  ${ONNXRUNTIME_ROOT}/core/providers/telum
  ${TEST_INC_DIR}
  ${ZDNN_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(onnxruntime_telum_test PRIVATE
  onnxruntime_providers_telum
  onnxruntime_test_utils
  onnxruntime_common
  onnxruntime_framework
  onnxruntime_graph
  onnxruntime_mlas
  GTest::gtest
  GTest::gtest_main
  ${ZDNN_LIB}
)

# Compiler definitions
target_compile_definitions(onnxruntime_telum_test PRIVATE
  TELUM_EP_TESTS
)

# Set C++ standard
set_target_properties(onnxruntime_telum_test PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Platform-specific settings
if(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
  # s390x specific flags
  target_compile_options(onnxruntime_telum_test PRIVATE
    -march=z16
    -mtune=z16
  )

  # Enable zDNN tests only on s390x
  target_compile_definitions(onnxruntime_telum_test PRIVATE
    ZDNN_AVAILABLE=1
  )
else()
  # On non-s390x platforms, tests will be skipped
  target_compile_definitions(onnxruntime_telum_test PRIVATE
    ZDNN_AVAILABLE=0
  )
endif()

# Add tests to CTest
add_test(
  NAME onnxruntime_telum_test
  COMMAND onnxruntime_telum_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Set test properties
set_tests_properties(onnxruntime_telum_test PROPERTIES
  LABELS "telum;provider"
  TIMEOUT 600
)

# Optional: Add test groups for selective execution
add_test(
  NAME onnxruntime_telum_test_utils
  COMMAND onnxruntime_telum_test --gtest_filter="TelumEndianTest.*"
)

add_test(
  NAME onnxruntime_telum_test_matmul
  COMMAND onnxruntime_telum_test --gtest_filter="TelumMatMulTest.*"
)

add_test(
  NAME onnxruntime_telum_test_gemm
  COMMAND onnxruntime_telum_test --gtest_filter="TelumGemmTest.*"
)

add_test(
  NAME onnxruntime_telum_test_elementwise
  COMMAND onnxruntime_telum_test --gtest_filter="TelumElementwiseTest.*"
)

add_test(
  NAME onnxruntime_telum_test_activations
  COMMAND onnxruntime_telum_test --gtest_filter="TelumActivationTest.*"
)

add_test(
  NAME onnxruntime_telum_test_softmax
  COMMAND onnxruntime_telum_test --gtest_filter="TelumSoftmaxTest.*"
)

add_test(
  NAME onnxruntime_telum_test_layernorm
  COMMAND onnxruntime_telum_test --gtest_filter="TelumLayerNormTest.*"
)

# Set labels for test groups
set_tests_properties(
  onnxruntime_telum_test_utils
  onnxruntime_telum_test_matmul
  onnxruntime_telum_test_gemm
  onnxruntime_telum_test_elementwise
  onnxruntime_telum_test_activations
  onnxruntime_telum_test_softmax
  onnxruntime_telum_test_layernorm
  PROPERTIES LABELS "telum;provider"
)

# Installation (optional)
if(onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_telum_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT test
  )
endif()

# Print configuration summary
message(STATUS "Telum EP Tests Configuration:")
message(STATUS "  Test source directory: ${TEST_SRC_DIR}")
message(STATUS "  Test include directory: ${TEST_INC_DIR}")
message(STATUS "  zDNN include directory: ${ZDNN_INCLUDE_DIR}")
message(STATUS "  zDNN library: ${ZDNN_LIB}")
message(STATUS "  Target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
  message(STATUS "  zDNN tests: ENABLED")
else()
  message(STATUS "  zDNN tests: DISABLED (not on s390x)")
endif()
