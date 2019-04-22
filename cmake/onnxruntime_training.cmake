# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# training lib
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


# training runner lib
file(GLOB_RECURSE onnxruntime_training_runner_srcs
    "${ONNXRUNTIME_ROOT}/test/training/runner/*.h"
    "${ONNXRUNTIME_ROOT}/test/training/runner/*.cc"
)
add_library(onnxruntime_training_runner ${onnxruntime_training_runner_srcs})
add_dependencies(onnxruntime_training_runner ${onnxruntime_EXTERNAL_DEPENDENCIES} onnx onnxruntime_providers)

onnxruntime_add_include_to_target(onnxruntime_training_runner  onnxruntime_common gsl onnx onnx_proto protobuf::libprotobuf onnxruntime_training)

if (onnxruntime_USE_CUDA)
target_include_directories(onnxruntime_training_runner PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} PUBLIC ${onnxruntime_graph_header} ${onnxruntime_CUDNN_HOME}/include)
else()
target_include_directories(onnxruntime_training_runner PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} PUBLIC ${onnxruntime_graph_header})
endif()

set_target_properties(onnxruntime_training_runner PROPERTIES FOLDER "ONNXRuntimeTest")
source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_training_runner_srcs})


# POC (MNIST)
file(GLOB_RECURSE training_poc_src
    "${ONNXRUNTIME_ROOT}/test/training/poc/*.h"
    "${ONNXRUNTIME_ROOT}/test/training/poc/mnist_data_provider.cc"
    "${ONNXRUNTIME_ROOT}/test/training/poc/main.cc"
)
add_executable(onnxruntime_training_poc ${training_poc_src})
onnxruntime_add_include_to_target(onnxruntime_training_poc onnxruntime_common gsl onnx onnx_proto protobuf::libprotobuf onnxruntime_training)
target_include_directories(onnxruntime_training_poc PUBLIC ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx onnxruntime_training_runner)

set(ONNXRUNTIME_LIBS onnxruntime_session ${onnxruntime_libs} ${PROVIDERS_CUDA} ${PROVIDERS_MKLDNN} onnxruntime_optimizer onnxruntime_providers onnxruntime_util onnxruntime_framework onnxruntime_util onnxruntime_graph onnxruntime_common onnxruntime_mlas)

target_link_libraries(onnxruntime_training_poc PRIVATE ${ONNXRUNTIME_LIBS} ${onnxruntime_EXTERNAL_LIBRARIES} onnxruntime_training onnxruntime_training_runner)
set_target_properties(onnxruntime_training_poc PROPERTIES FOLDER "ONNXRuntimeTest")


# squeezenet
file(GLOB_RECURSE training_squeezene_src
    "${ONNXRUNTIME_ROOT}/test/training/squeezenet/*.h"
    "${ONNXRUNTIME_ROOT}/test/training/squeezenet/*.cc"
)
add_executable(onnxruntime_training_squeezenet ${training_squeezene_src})
onnxruntime_add_include_to_target(onnxruntime_training_squeezenet onnxruntime_common gsl onnx onnx_proto protobuf::libprotobuf onnxruntime_training)
target_include_directories(onnxruntime_training_squeezenet PUBLIC ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${extra_includes} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx onnxruntime_training_runner)

target_link_libraries(onnxruntime_training_squeezenet PRIVATE ${ONNXRUNTIME_LIBS} ${onnxruntime_EXTERNAL_LIBRARIES} onnxruntime_training onnxruntime_training_runner)
set_target_properties(onnxruntime_training_squeezenet PROPERTIES FOLDER "ONNXRuntimeTest")
