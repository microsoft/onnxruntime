# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_training_srcs
    "${ONNXRUNTIME_ROOT}/core/training/*.h"
    "${ONNXRUNTIME_ROOT}/core/training/*.cc"
)

add_library(onnxruntime_training ${onnxruntime_training_srcs})
add_dependencies(onnxruntime_training ${onnxruntime_EXTERNAL_DEPENDENCIES} onnx)
onnxruntime_add_include_to_target(onnxruntime_training  onnxruntime_common gsl onnx onnx_proto protobuf::libprotobuf)
target_include_directories(onnxruntime_training PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} PUBLIC ${onnxruntime_graph_header})
set_target_properties(onnxruntime_training PROPERTIES FOLDER "ONNXRuntime")
source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_training_srcs})

# POC
set(training_test_src_dir ${ONNXRUNTIME_ROOT}/test/training)
add_executable(onnxruntime_training_poc ${training_test_src_dir}/poc/main.cc)
onnxruntime_add_include_to_target(onnxruntime_training_poc gsl onnx onnx_proto protobuf::libprotobuf)
target_include_directories(onnxruntime_training_poc PUBLIC ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx)

set(ONNXRUNTIME_LIBS
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MKLDNN}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
)

target_link_libraries(onnxruntime_training_poc PRIVATE ${ONNXRUNTIME_LIBS} ${onnxruntime_EXTERNAL_LIBRARIES})
set_target_properties(onnxruntime_training_poc PROPERTIES FOLDER "ONNXRuntimeTest")