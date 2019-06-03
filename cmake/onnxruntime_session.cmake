# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_session_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/session/*.h"
    "${ONNXRUNTIME_ROOT}/core/session/*.h"
    "${ONNXRUNTIME_ROOT}/core/session/*.cc"
    )

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_session_srcs})

add_library(onnxruntime_session ${onnxruntime_session_srcs})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/session  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
onnxruntime_add_include_to_target(onnxruntime_session onnxruntime_common onnxruntime_framework gsl onnx onnx_proto protobuf::libprotobuf)
target_include_directories(onnxruntime_session PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
add_dependencies(onnxruntime_session ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_session PROPERTIES FOLDER "ONNXRuntime")

if(onnxruntime_USE_EIGEN_THREADPOOL)
    target_compile_definitions(onnxruntime_session PUBLIC USE_EIGEN_THREADPOOL)
endif()
