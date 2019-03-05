# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

project(onnxruntime_hosting)

find_package(Boost 1.68 COMPONENTS system coroutine context thread program_options REQUIRED)

file(GLOB_RECURSE onnxruntime_hosting_srcs
    "${ONNXRUNTIME_ROOT}/hosting/*.h"
    "${ONNXRUNTIME_ROOT}/hosting/*.cc"
)

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_hosting_srcs})

add_executable(${PROJECT_NAME} ${onnxruntime_hosting_srcs})

onnxruntime_add_include_to_target(${PROJECT_NAME} onnxruntime_session gsl)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${ONNXRUNTIME_ROOT}
        ${Boost_INCLUDE_DIR}
        PUBLIC
        ${ONNXRUNTIME_ROOT}/hosting/include)

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${Boost_LIBRARIES}
        onnxruntime_session
        onnxruntime_optimizer
        onnxruntime_providers
        onnxruntime_util
        onnxruntime_framework
        onnxruntime_util
        onnxruntime_graph
        onnxruntime_common
        onnxruntime_mlas
        onnx
        onnx_proto
        protobuf::libprotobuf
        re2
)

