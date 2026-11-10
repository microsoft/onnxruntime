# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT DEFINED TEST_WORK_DIR OR NOT DEFINED ORT_CMAKE_DIR)
  message(FATAL_ERROR "TEST_WORK_DIR and ORT_CMAKE_DIR are required")
endif()

set(include_root "${TEST_WORK_DIR}/include_fixture")
file(MAKE_DIRECTORY "${include_root}/core")
foreach(include IN ITEMS "stateful/session/session_loop.h" "../../stateful/session/session_loop.h")
  file(WRITE "${include_root}/core/example.cc" "#include \"${include}\"\n")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" "-DONNXRUNTIME_ROOT=${include_root}"
            -P "${ORT_CMAKE_DIR}/verify_stateful_includes.cmake"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(result EQUAL 0 OR NOT error MATCHES "Core source includes a private stateful header")
    message(FATAL_ERROR "Forbidden include was not detected: ${include}\n${output}\n${error}")
  endif()
endforeach()
file(WRITE "${include_root}/core/example.cc" "#include \"core/session/ort_apis.h\"\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" "-DONNXRUNTIME_ROOT=${include_root}"
          -P "${ORT_CMAKE_DIR}/verify_stateful_includes.cmake"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Allowed include was rejected:\n${output}\n${error}")
endif()

set(public_include_root "${include_root}/public")
file(MAKE_DIRECTORY "${public_include_root}/core")
file(WRITE "${public_include_root}/core/example.h" "#include <stateful/session/session_loop.h>\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" "-DONNXRUNTIME_ROOT=${include_root}"
          "-DONNXRUNTIME_INCLUDE_ROOT=${public_include_root}"
          -P "${ORT_CMAKE_DIR}/verify_stateful_includes.cmake"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(result EQUAL 0 OR NOT error MATCHES "Core source includes a private stateful header")
  message(FATAL_ERROR "Forbidden public include was not detected:\n${output}\n${error}")
endif()

set(graph_root "${TEST_WORK_DIR}/graph_fixture")
file(MAKE_DIRECTORY "${graph_root}")
file(WRITE "${graph_root}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.24)
project(StatefulBoundaryFixture NONE)
add_library(onnxruntime_stateful INTERFACE)
add_library(stateful_alias ALIAS onnxruntime_stateful)
add_library(intermediate INTERFACE)
add_library(core INTERFACE)
target_link_libraries(core INTERFACE intermediate)
if(TEST_FORBIDDEN STREQUAL "ALIAS")
  target_link_libraries(intermediate INTERFACE "$<LINK_ONLY:stateful_alias>")
elseif(TEST_FORBIDDEN)
  target_link_libraries(intermediate INTERFACE "$<LINK_ONLY:onnxruntime_stateful>")
endif()
include("${BOUNDARY_CHECKER}")
onnxruntime_check_stateful_target_boundary(core)
]=])
foreach(mode IN ITEMS OFF ON ALIAS)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${graph_root}" -B "${graph_root}/build_${mode}"
            "-DBOUNDARY_CHECKER=${ORT_CMAKE_DIR}/verify_stateful_target_boundary.cmake"
            "-DTEST_FORBIDDEN=${mode}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(mode STREQUAL "OFF" AND NOT result EQUAL 0)
    message(FATAL_ERROR "Allowed dependency was rejected:\n${output}\n${error}")
  elseif(NOT mode STREQUAL "OFF" AND
         (result EQUAL 0 OR NOT error MATCHES "Stateful build boundary violated"))
    message(FATAL_ERROR "Forbidden transitive dependency was not detected:\n${output}\n${error}")
  endif()
endforeach()
