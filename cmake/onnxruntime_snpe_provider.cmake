# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(PROVIDERS_SNPE onnxruntime_providers_snpe)
add_compile_definitions(USE_SNPE=1)

file(GLOB_RECURSE
  onnxruntime_providers_snpe_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/snpe/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/snpe/*.cc"
)

if(ANDROID)
  # Specify the link libraries
  set(SNPE_NN_LIBS ${SNPE} libc++_shared.so)
else()
  set(SNPE_NN_LIBS ${SNPE})
endif()

file(TO_CMAKE_PATH ${SNPE_ROOT} SNPE_ROOT)
get_filename_component(SNPE_CMAKE_DIR ${SNPE_ROOT} ABSOLUTE)
file(TO_CMAKE_PATH "${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI}" SNPE_LIB_DIR)
file(TO_NATIVE_PATH ${SNPE_LIB_DIR} SNPE_NATIVE_DIR)
message(STATUS "Looking for SNPE library in ${SNPE_NATIVE_DIR}")
find_library(SNPE NAMES snpe SNPE libSNPE.so PATHS "${SNPE_NATIVE_DIR}" "${SNPE_ROOT}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
file(GLOB SNPE_SO_FILES LIST_DIRECTORIES false "${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI}/*.so" "${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI}/*.dll")
# add dsp skel files to distribution
file(GLOB SNPE_DSP_FILES LIST_DIRECTORIES false "${SNPE_CMAKE_DIR}/lib/dsp/*.so")
list(APPEND SNPE_SO_FILES ${QCDK_FILES} ${SNPE_DSP_FILES})

if(NOT SNPE OR NOT SNPE_SO_FILES)
  message(ERROR "Snpe not found in ${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI} for platform ${CMAKE_GENERATOR_PLATFORM}")
endif()
message(STATUS "SNPE library at ${SNPE}")
message(STATUS "SNPE so/dlls in ${SNPE_SO_FILES}")

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_snpe_cc_srcs})
onnxruntime_add_static_library(onnxruntime_providers_snpe ${onnxruntime_providers_snpe_cc_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_snpe onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf-lite flatbuffers)
link_directories(${SNPE_CMAKE_DIR}/lib/${SNPE_ARCH_ABI} ${SNPE_CMAKE_DIR}/lib/dsp)
target_link_libraries(onnxruntime_providers_snpe PRIVATE SNPE ${SNPE_NN_LIBS})
add_dependencies(onnxruntime_providers_snpe onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_providers_snpe PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(onnxruntime_providers_snpe PROPERTIES FOLDER "ONNXRuntime")
target_include_directories(onnxruntime_providers_snpe PRIVATE ${ONNXRUNTIME_ROOT} ${SNPE_ROOT}/include/zdl)
set_target_properties(onnxruntime_providers_snpe PROPERTIES LINKER_LANGUAGE CXX)

if(MSVC)
  target_compile_options(onnxruntime_providers_snpe PUBLIC /wd4244 /wd4505)
endif()
