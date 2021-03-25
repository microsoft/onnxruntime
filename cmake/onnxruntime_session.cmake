# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_session_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/session/*.h"
    "${ONNXRUNTIME_ROOT}/core/session/*.h"
    "${ONNXRUNTIME_ROOT}/core/session/*.cc"
    "${ONNXRUNTIME_ROOT}/core/session/pipeline_parallelism/*.h"
    "${ONNXRUNTIME_ROOT}/core/session/pipeline_parallelism/*.cc"
    )

if (onnxruntime_USE_CUDA)
  file(GLOB onnxruntime_pipeline_parallelism_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/session/pipeline_parallelism/*.h"
    "${ONNXRUNTIME_ROOT}/core/session/pipeline_parallelism/*.cc"
    )
  list (APPEND onnxruntime_session_srcs ${onnxruntime_pipeline_parallelism_srcs})
endif()

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_session_srcs})

add_library(onnxruntime_session ${onnxruntime_session_srcs})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/session  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
onnxruntime_add_include_to_target(onnxruntime_session onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
if(onnxruntime_ENABLE_INSTRUMENT)
  target_compile_definitions(onnxruntime_session PUBLIC ONNXRUNTIME_ENABLE_INSTRUMENT)
endif()
target_include_directories(onnxruntime_session PRIVATE ${ONNXRUNTIME_ROOT} ${PROJECT_SOURCE_DIR}/external/json ${eigen_INCLUDE_DIRS})
add_dependencies(onnxruntime_session ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_session PROPERTIES FOLDER "ONNXRuntime")
if (onnxruntime_USE_CUDA)
  target_include_directories(onnxruntime_session PRIVATE ${onnxruntime_CUDNN_HOME}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
if (onnxruntime_ENABLE_TRAINING)
  target_include_directories(onnxruntime_session PRIVATE ${ORTTRAINING_ROOT})
endif()
