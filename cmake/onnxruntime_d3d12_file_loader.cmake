# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include_guard(GLOBAL)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "onnxruntime_d3d12_file_loader requires native Windows MSVC.")
endif()

set(onnxruntime_d3d12_file_loader_srcs
  "${ONNXRUNTIME_ROOT}/core/platform/windows/d3d12_file_loader/d3d12_file_buffer_loader.h"
  "${ONNXRUNTIME_ROOT}/core/platform/windows/d3d12_file_loader/d3d12_file_buffer_loader.cc"
)

source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_d3d12_file_loader_srcs})
onnxruntime_add_static_library(
  onnxruntime_d3d12_file_loader
  ${onnxruntime_d3d12_file_loader_srcs})
onnxruntime_add_include_to_target(
  onnxruntime_d3d12_file_loader
  onnxruntime_common ${WIL_TARGET} ${GSL_TARGET})
target_link_libraries(
  onnxruntime_d3d12_file_loader
  PUBLIC d3d12.lib dxgi.lib dxguid.lib
  PRIVATE ${WIL_TARGET} ${GSL_TARGET})
set_target_properties(
  onnxruntime_d3d12_file_loader
  PROPERTIES FOLDER "ONNXRuntime")
