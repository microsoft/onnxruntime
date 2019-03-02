# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${ONNXRUNTIME_ROOT}/cmake/external/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

project(onnxruntime_hosting)
set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_NAME}" PRIVATE ${ONNXRUNTIME_ROOT}/hosting/include")

file(GLOB_RECURSE onnxruntime_hosting_srcs
    "${ONNXRUNTIME_ROOT}/hosting/*.h"
    "${ONNXRUNTIME_ROOT}/hosting/*.cc"
)

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_hosting_srcs})

add_executable(${PROJECT_NAME} ${onnxruntime_hosting_srcs})

onnxruntime_add_include_to_target(${PROJECT_NAME} onnxruntime_session gsl)
target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_ROOT})

target_link_libraries(${PROJECT_NAME} PRIVATE
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

