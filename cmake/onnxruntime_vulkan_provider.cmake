# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(PROVIDERS_VULKAN onnxruntime_providers_vulkan)
add_compile_definitions(USE_VULKAN=1)

file(GLOB_RECURSE
  onnxruntime_providers_vulkan_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/vulkan/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/vulkan/*.cc"
)

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_vulkan_cc_srcs})
onnxruntime_add_static_library(onnxruntime_providers_vulkan ${onnxruntime_providers_vulkan_cc_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_vulkan onnxruntime_common onnxruntime_framework onnx onnx_proto)

set_target_properties(onnxruntime_providers_vulkan PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(onnxruntime_providers_vulkan PROPERTIES FOLDER "ONNXRuntime")
set_target_properties(onnxruntime_providers_vulkan PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(onnxruntime_providers_vulkan onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})

link_directories(${VULKAN_HOME}/Lib)
target_link_libraries(onnxruntime_providers_vulkan PRIVATE vulkan-1)
target_include_directories(onnxruntime_providers_vulkan PRIVATE ${ONNXRUNTIME_ROOT} ${VULKAN_HOME}/Include)