# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.28)
project(cuda_architectures_test NONE)

if(NOT DEFINED TEST_CUDA_ARCHITECTURES OR NOT DEFINED EXPECTED_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "TEST_CUDA_ARCHITECTURES and EXPECTED_CUDA_ARCHITECTURES are required")
endif()

set(CMAKE_CUDA_ARCHITECTURES "${TEST_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures to test")
set(CMAKE_CUDA_COMPILER_VERSION 13.0)

include("${CMAKE_CURRENT_LIST_DIR}/../../external/cuda_configuration.cmake")
setup_cuda_architectures()

if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "${EXPECTED_CUDA_ARCHITECTURES}")
  message(
    FATAL_ERROR
      "Expected CUDA architectures '${EXPECTED_CUDA_ARCHITECTURES}', got '${CMAKE_CUDA_ARCHITECTURES}'")
endif()
