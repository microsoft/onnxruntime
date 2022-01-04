# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(onnxruntime_optimizer_src_patterns)

if (onnxruntime_MINIMAL_BUILD)
  # we include a couple of files so a library is produced and we minimize other changes to the build setup.
  # if the transformer base class is unused it will be excluded from the final binary size
  list(APPEND onnxruntime_optimizer_src_patterns
    "${ONNXRUNTIME_INCLUDE_DIR}/core/optimizer/graph_transformer.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/graph_transformer.cc"
  )

  if (onnxruntime_ENABLE_RUNTIME_OPTIMIZATION_REPLAY_IN_MINIMAL_BUILD)
    list(APPEND onnxruntime_optimizer_src_patterns
      "${ONNXRUNTIME_INCLUDE_DIR}/core/optimizer/graph_transformer_utils.h"
      "${ONNXRUNTIME_ROOT}/core/optimizer/graph_transformer_utils.cc"
      "${ONNXRUNTIME_ROOT}/core/optimizer/qdq_transformer/qdq_util.h"
      "${ONNXRUNTIME_ROOT}/core/optimizer/qdq_transformer/qdq_util.cc"
      "${ONNXRUNTIME_ROOT}/core/optimizer/qdq_transformer/selectors_actions/*.h"
      "${ONNXRUNTIME_ROOT}/core/optimizer/qdq_transformer/selectors_actions/*.cc"
      "${ONNXRUNTIME_ROOT}/core/optimizer/selectors_actions/*.h"
      "${ONNXRUNTIME_ROOT}/core/optimizer/selectors_actions/*.cc"
    )
  endif()
else()
  list(APPEND onnxruntime_optimizer_src_patterns
    "${ONNXRUNTIME_INCLUDE_DIR}/core/optimizer/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/*.cc"
    "${ONNXRUNTIME_ROOT}/core/optimizer/qdq_transformer/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/qdq_transformer/*.cc"
    "${ONNXRUNTIME_ROOT}/core/optimizer/qdq_transformer/selectors_actions/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/qdq_transformer/selectors_actions/*.cc"
    "${ONNXRUNTIME_ROOT}/core/optimizer/selectors_actions/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/selectors_actions/*.cc"
    "${ONNXRUNTIME_ROOT}/core/optimizer/transpose_optimizer/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/transpose_optimizer/*.cc"
  )
endif()

if (onnxruntime_ENABLE_TRAINING)
  list(APPEND onnxruntime_optimizer_src_patterns
    "${ORTTRAINING_SOURCE_DIR}/core/optimizer/*.h"
    "${ORTTRAINING_SOURCE_DIR}/core/optimizer/*.cc"
  )
endif()

file(GLOB onnxruntime_optimizer_srcs CONFIGURE_DEPENDS ${onnxruntime_optimizer_src_patterns})

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_optimizer_srcs})

if (onnxruntime_EXTERNAL_TRANSFORMER_SRC_PATH)
  set(onnxruntime_external_transformer_src_patterns)
  list(APPEND onnxruntime_external_transformer_src_patterns
    "${onnxruntime_EXTERNAL_TRANSFORMER_SRC_PATH}/*.cc"
    "${onnxruntime_EXTERNAL_TRANSFORMER_SRC_PATH}/*.cpp"
  )
  file(GLOB onnxruntime_external_transformer_src ${onnxruntime_external_transformer_src_patterns})
  list(APPEND onnxruntime_optimizer_srcs ${onnxruntime_external_transformer_src})
endif()

onnxruntime_add_static_library(onnxruntime_optimizer ${onnxruntime_optimizer_srcs})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/optimizer  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
onnxruntime_add_include_to_target(onnxruntime_optimizer onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)
target_include_directories(onnxruntime_optimizer PRIVATE ${ONNXRUNTIME_ROOT})
if (onnxruntime_ENABLE_TRAINING)
  target_include_directories(onnxruntime_optimizer PRIVATE ${ORTTRAINING_ROOT})
endif()
add_dependencies(onnxruntime_optimizer ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_optimizer PROPERTIES FOLDER "ONNXRuntime")
