# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
cmake_minimum_required(VERSION 3.26)
project(phi_allocation LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
foreach(required ORT_HOME GENAI_SOURCE_DIR GENAI_LIBRARY_DIR CUDAToolkit_ROOT)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "${required} is required (no machine-specific fallback)")
  endif()
endforeach()
find_package(CUDAToolkit 13 REQUIRED)
foreach(phase a b)
  add_executable(phi_phase_${phase} phase_${phase}.cpp)
  target_include_directories(phi_phase_${phase} PRIVATE "${ORT_HOME}/include" "${GENAI_SOURCE_DIR}/src")
  target_link_libraries(phi_phase_${phase} PRIVATE
    "${ORT_HOME}/lib/libonnxruntime.so" "${GENAI_LIBRARY_DIR}/libonnxruntime-genai.so")
  set_target_properties(phi_phase_${phase} PROPERTIES
    BUILD_RPATH "${ORT_HOME}/lib;${GENAI_LIBRARY_DIR};${CUDAToolkit_LIBRARY_DIR}")
  target_compile_options(phi_phase_${phase} PRIVATE -Wall -Wextra)
  target_link_options(phi_phase_${phase} PRIVATE "-Wl,--enable-new-dtags")
endforeach()
target_link_libraries(phi_phase_b PRIVATE CUDA::cudart)
