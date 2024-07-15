# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# set env var that cmake uses for the Vulkan SDK PATH
# onnxruntime_external_deps.cmake should have set this up
#
# set(ENV{VULKAN_SDK} ${onnxruntime_VULKAN_SDK_PATH})
# find_package(Vulkan REQUIRED)

if (NOT VULKAN_FOUND)
  message(FATAL_ERROR "Vulkan SDK was not found. onnxruntime_VULKAN_SDK_PATH is set to ${onnxruntime_VULKAN_SDK_PATH}")
endif()

message(STATUS "Vulkan_INCLUDE_DIRS: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan_LIBRARIES: ${Vulkan_LIBRARIES}")
# message(STATUS "Vulkan_dxc_EXECUTABLE: ${Vulkan_dxc_EXECUTABLE}")

add_compile_definitions(USE_VULKAN=1)
file(GLOB_RECURSE onnxruntime_providers_vulkan_cc_srcs
  "${ONNXRUNTIME_ROOT}/core/providers/vulkan/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/vulkan/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/vulkan/math/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/vulkan/math/*.cc"
)

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_vulkan_cc_srcs})
onnxruntime_add_static_library(onnxruntime_providers_vulkan ${onnxruntime_providers_vulkan_cc_srcs})

onnxruntime_add_include_to_target(onnxruntime_providers_vulkan
  onnxruntime_common onnxruntime_framework
  ncnn Vulkan::Headers
  onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface
)

target_include_directories(onnxruntime_providers_vulkan PRIVATE "$<TARGET_PROPERTY:ncnn,SOURCE_DIR>/layer")
target_include_directories(onnxruntime_providers_vulkan PRIVATE "$<TARGET_PROPERTY:ncnn,SOURCE_DIR>/layer/vulkan")

target_link_libraries(onnxruntime_providers_vulkan ncnn Vulkan::Vulkan)
add_dependencies(onnxruntime_providers_vulkan ${onnxruntime_EXTERNAL_DEPENDENCIES})

set_target_properties(onnxruntime_providers_vulkan PROPERTIES FOLDER "ONNXRuntime")
set_target_properties(onnxruntime_providers_vulkan PROPERTIES LINKER_LANGUAGE CXX)

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_vulkan
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
