# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_eager_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/eager/*.h"
    "${ONNXRUNTIME_ROOT}/core/eager/*.cc"
    )

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_eager_srcs})

add_library(onnxruntime_eager ${onnxruntime_eager_srcs})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/eager  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
onnxruntime_add_include_to_target(onnxruntime_eager onnxruntime_common onnxruntime_framework onnxruntime_optimizer onnxruntime_graph onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)
if(onnxruntime_ENABLE_INSTRUMENT)
  target_compile_definitions(onnxruntime_eager PUBLIC ONNXRUNTIME_ENABLE_INSTRUMENT)
endif()
target_include_directories(onnxruntime_eager PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
add_dependencies(onnxruntime_eager ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_eager PROPERTIES FOLDER "ONNXRuntime")
if (onnxruntime_ENABLE_TRAINING)
  target_include_directories(onnxruntime_session PRIVATE ${ORTTRAINING_ROOT})
endif()
