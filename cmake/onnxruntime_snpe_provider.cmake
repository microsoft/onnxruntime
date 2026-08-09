# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(PROVIDERS_SNPE onnxruntime_providers_snpe)
add_compile_definitions(USE_SNPE=1)

file(GLOB_RECURSE
  onnxruntime_providers_snpe_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/snpe/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/snpe/*.cc"
)

file(GLOB SNPE_SO_FILES LIST_DIRECTORIES false "${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI}/*.so" "${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI}/*.dll")
# add dsp skel files to distribution
file(GLOB SNPE_DSP_FILES LIST_DIRECTORIES false "${SNPE_CMAKE_DIR}/lib/dsp/*.so")
list(APPEND SNPE_SO_FILES ${QCDK_FILES} ${SNPE_DSP_FILES})

if(NOT SNPE OR NOT SNPE_SO_FILES)
  message(ERROR "Snpe not found in ${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI} for platform ${CMAKE_GENERATOR_PLATFORM}")
endif()

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_snpe_cc_srcs})
onnxruntime_add_static_library(onnxruntime_providers_snpe ${onnxruntime_providers_snpe_cc_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_snpe onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf-lite flatbuffers::flatbuffers)
link_directories(${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI} ${SNPE_CMAKE_DIR}/lib/dsp)
add_dependencies(onnxruntime_providers_snpe onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_providers_snpe PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(onnxruntime_providers_snpe PROPERTIES FOLDER "ONNXRuntime")
target_include_directories(onnxruntime_providers_snpe PRIVATE ${ONNXRUNTIME_ROOT} ${SNPE_ROOT}/include/zdl)
set_target_properties(onnxruntime_providers_snpe PROPERTIES LINKER_LANGUAGE CXX)

if(MSVC)
  target_compile_options(onnxruntime_providers_snpe PUBLIC /wd4244 /wd4505)
endif()
