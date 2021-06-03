# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Schema library (for contrib ops and training ops)

set (CONTRIB_OPS_DIR ${ONNXRUNTIME_ROOT}/core/graph/contrib_ops)
set (TRAINING_OPS_DIR ${ORTTRAINING_ROOT}/orttraining/core/graph)

file(GLOB_RECURSE contrib_ops_schema_src
   "${CONTRIB_OPS_DIR}/*.cc"
   "${TRAINING_OPS_DIR}/training_op_defs.cc"
)

add_library(ort_opschema_lib ${contrib_ops_schema_src})

set (OPSCHEMA_LIB_DEPENDENCIES onnxruntime_mlas onnxruntime_common onnxruntime_util onnx onnx_proto protobuf::libprotobuf flatbuffers)

# ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" is found
target_include_directories(ort_opschema_lib PRIVATE ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(ort_opschema_lib ${OPSCHEMA_LIB_DEPENDENCIES})
onnxruntime_add_include_to_target(ort_opschema_lib onnxruntime_common onnx onnx_proto protobuf::libprotobuf flatbuffers)

# Test schema library using toy application

set(OPSCHEMA_LIB_TEST ${REPO_ROOT}/samples/c_cxx/opschema_lib_use)

file(GLOB_RECURSE opschema_lib_test_src "${OPSCHEMA_LIB_TEST}/*.cc")

add_executable(opschema_lib_test ${opschema_lib_test_src})

target_include_directories(opschema_lib_test PRIVATE ${ORTTRAINING_ROOT})

target_link_libraries(opschema_lib_test ort_opschema_lib ${OPSCHEMA_LIB_DEPENDENCIES}) 
