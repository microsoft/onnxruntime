# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(TENSORBOARD_ROOT ${PROJECT_SOURCE_DIR}/external/tensorboard)

# tensorboard protos
file(GLOB_RECURSE tensorboard_proto_srcs CONFIGURE_DEPENDS
  "${TENSORBOARD_ROOT}/tensorboard/compat/proto/*.proto"
)

if(EXISTS "${ONNX_CUSTOM_PROTOC_EXECUTABLE}")
  set(PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
else()
  set(PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
  set(PROTOC_DEPS protobuf::protoc)
endif()
if(NOT onnxruntime_USE_FULL_PROTOBUF)
  set(PROTOC_PROTOBUF_ARG "lite:")
endif()

# tensorboard generated files
foreach(_proto ${tensorboard_proto_srcs})
  get_filename_component(_abs_file ${_proto} ABSOLUTE)
  get_filename_component(_basename ${_proto} NAME_WE)
  set(_tensorboard_cpp_srcs
    "${CMAKE_CURRENT_BINARY_DIR}/tensorboard/compat/proto/${_basename}.pb.cc"
    "${CMAKE_CURRENT_BINARY_DIR}/tensorboard/compat/proto/${_basename}.pb.h"
  )
  add_custom_command(
    OUTPUT ${_tensorboard_cpp_srcs}
    COMMAND ${PROTOC_EXECUTABLE}
    ARGS --cpp_out ${PROTOC_PROTOBUF_ARG}${CMAKE_CURRENT_BINARY_DIR} -I ${TENSORBOARD_ROOT} -I ${REPO_ROOT}/cmake/external/protobuf/src ${_abs_file}
    DEPENDS ${_abs_file} ${PROTOC_DEPS}
    COMMENT "Running cpp protocol buffer compiler on ${_proto}"
    VERBATIM )
  list(APPEND tensorboard_cpp_srcs ${_tensorboard_cpp_srcs})
endforeach()

add_library(tensorboard ${tensorboard_cpp_srcs})
target_include_directories(tensorboard PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(tensorboard PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_COMPILE_DEFINITIONS>)

if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    target_compile_options(tensorboard PRIVATE "-Wno-unused-parameter")
  endif()
  if(HAS_UNUSED_BUT_SET_VARIABLE)
    target_compile_options(tensorboard PRIVATE "-Wno-unused-but-set-variable")
  endif()
endif()
if(UNIX)
  target_compile_options(tensorboard PRIVATE "-Wno-unused-variable")
  target_compile_options(tensorboard PRIVATE "-Wno-deprecated-declarations")
endif()

if(WIN32)
  target_compile_options(tensorboard PRIVATE /wd4100 /wd4996 /wd4244 /wd4267 /wd4309)
endif()


set_target_properties(tensorboard PROPERTIES FOLDER "External/tensorboard")
