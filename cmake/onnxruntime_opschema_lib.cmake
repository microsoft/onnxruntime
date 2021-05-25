# Schema library (for contrib ops and training ops)

set (CONTRIB_OPS_DIR ${ONNXRUNTIME_ROOT}/core/graph/contrib_ops)
set (TRAINING_OPS_DIR ${ORTTRAINING_ROOT}/orttraining/core/graph)

file(GLOB_RECURSE contrib_ops_schema_src
   "${CONTRIB_OPS_DIR}/*.cc"
   "${TRAINING_OPS_DIR}/training_op_defs.cc"
)

add_library(ort_opschema_lib ${contrib_ops_schema_src})

target_include_directories(ort_opschema_lib PRIVATE ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${CMAKE_BINARY_DIR})
onnxruntime_add_include_to_target(ort_opschema_lib onnxruntime_common onnx onnx_proto protobuf::libprotobuf flatbuffers)

set (OPSCHEMA_LIB_DEPENDENCIES onnxruntime_mlas onnxruntime_common onnxruntime_util onnx onnx_proto protobuf::libprotobuf flatbuffers)

# Test schema library using toy application

set(OPSCHEMA_LIB_TEST ${ORTTRAINING_ROOT}/tools/opschema_lib_test)

file(GLOB_RECURSE opschema_lib_test_src "${OPSCHEMA_LIB_TEST}/*.cc")

add_executable(opschema_lib_test ${opschema_lib_test_src})

target_include_directories(opschema_lib_test ${ORTTRAINING_ROOT})
target_link_libraries(opschema_lib_test ort_opschema_lib ${OPSCHEMA_LIB_DEPENDENCIES}) 
