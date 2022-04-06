# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(_onnxflow_pb_cpp_srcs
    "${ORTTRAINING_ROOT}/orttraining/onnxflow/csrc/onnxflow.pb.cc"
    "${ORTTRAINING_ROOT}/orttraining/onnxflow/csrc/onnxflow.pb.h"
  )

if(EXISTS "${ONNX_CUSTOM_PROTOC_EXECUTABLE}")
  set(PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
else()
  set(PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
  set(PROTOC_DEPS protobuf::protoc)
endif()

add_custom_command(
    OUTPUT ${_onnxflow_pb_cpp_srcs}
    COMMAND ${PROTOC_EXECUTABLE}
    ARGS --python_out=${ORTTRAINING_ROOT}/orttraining/onnxflow/onnxflow/ --cpp_out=${ORTTRAINING_ROOT}/orttraining/onnxflow/csrc/ --proto_path=${ORTTRAINING_ROOT}/orttraining/onnxflow --proto_path=${REPO_ROOT}/cmake/external/protobuf/src ${ORTTRAINING_ROOT}/orttraining/onnxflow/onnxflow.proto
    DEPENDS ${ORTTRAINING_ROOT}/orttraining/onnxflow/onnxflow.proto ${PROTOC_DEPS}
    COMMENT "Running cpp protocol buffer compiler on onnxflow.proto"
    VERBATIM )

file(GLOB onnxruntime_on_device_training_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_ROOT}/orttraining/onnxflow/csrc/*.h"
    "${ORTTRAINING_ROOT}/orttraining/onnxflow/csrc/*.cpp"
    )
list(APPEND onnxruntime_on_device_training_srcs ${_onnxflow_pb_cpp_srcs})

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_on_device_training_srcs})

onnxruntime_add_static_library(onnxruntime_on_device_training ${onnxruntime_on_device_training_srcs})

onnxruntime_add_include_to_target(onnxruntime_on_device_training onnxruntime_common onnxruntime_framework onnxruntime_optimizer onnxruntime_graph onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)
target_include_directories(onnxruntime_on_device_training PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
add_dependencies(onnxruntime_on_device_training ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_on_device_training PROPERTIES FOLDER "ONNXRuntime")
if (onnxruntime_ENABLE_TRAINING)
  target_include_directories(onnxruntime_session PRIVATE ${ORTTRAINING_ROOT})
endif()

# sample loading of file
file(GLOB orttraining_on_device_sample_src CONFIGURE_DEPENDS
    "${ORTTRAINING_ROOT}/orttraining/onnxflow/sample.m.cpp"
    )
onnxruntime_add_executable(orttraining_on_device_sample ${orttraining_on_device_sample_src})
onnxruntime_add_include_to_target(orttraining_on_device_sample onnxruntime_on_device_training onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} onnxruntime_training flatbuffers)
target_include_directories(orttraining_on_device_sample PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${eigen_INCLUDE_DIRS} ${CXXOPTS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx onnxruntime_training_runner ${PROTOBUF_LIB})

target_link_libraries(orttraining_on_device_sample PRIVATE onnxruntime_on_device_training onnx onnx_proto onnxruntime_training ${ONNXRUNTIME_LIBS} ${onnxruntime_EXTERNAL_LIBRARIES} libprotobuf)
# set_target_properties(onnxruntime_training_mnist PROPERTIES FOLDER "ONNXRuntimeTest")

