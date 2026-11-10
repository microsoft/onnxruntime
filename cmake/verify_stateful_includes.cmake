# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT DEFINED ONNXRUNTIME_ROOT OR NOT IS_DIRECTORY "${ONNXRUNTIME_ROOT}/core")
  message(FATAL_ERROR "ONNXRUNTIME_ROOT must point to the onnxruntime source directory")
endif()

set(core_roots "${ONNXRUNTIME_ROOT}/core")
if(DEFINED ONNXRUNTIME_INCLUDE_ROOT)
  list(APPEND core_roots "${ONNXRUNTIME_INCLUDE_ROOT}/core")
endif()
set(core_sources)
foreach(core_root IN LISTS core_roots)
  file(GLOB_RECURSE sources LIST_DIRECTORIES false
    "${core_root}/*.c"
    "${core_root}/*.cc"
    "${core_root}/*.cpp"
    "${core_root}/*.cu"
    "${core_root}/*.cuh"
    "${core_root}/*.h"
    "${core_root}/*.hpp"
    "${core_root}/*.m"
    "${core_root}/*.mm")
  list(APPEND core_sources ${sources})
endforeach()
foreach(source IN LISTS core_sources)
  file(STRINGS "${source}" forbidden_includes
    REGEX "^[ \t]*#[ \t]*include[ \t]*[<\"]([^\">]*[/\\\\])?stateful[/\\\\]")
  if(forbidden_includes)
    message(FATAL_ERROR "Core source includes a private stateful header: ${source}: ${forbidden_includes}")
  endif()
endforeach()
