# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(CXXOPTS ${cxxopts_SOURCE_DIR}/include)

# training lib
file(GLOB_RECURSE onnxruntime_training_srcs
  "${ORTTRAINING_SOURCE_DIR}/core/framework/*.h"
  "${ORTTRAINING_SOURCE_DIR}/core/framework/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/core/framework/tensorboard/*.h"
  "${ORTTRAINING_SOURCE_DIR}/core/framework/tensorboard/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/core/framework/adasum/*"
  "${ORTTRAINING_SOURCE_DIR}/core/framework/communication/*"
  "${ORTTRAINING_SOURCE_DIR}/core/agent/*.h"
  "${ORTTRAINING_SOURCE_DIR}/core/agent/*.cc"
)

# This needs to be built in framework.cmake
file(GLOB_RECURSE onnxruntime_training_framework_excluded_srcs CONFIGURE_DEPENDS
  "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/*.h"
  "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/core/framework/triton/*.h"
  "${ORTTRAINING_SOURCE_DIR}/core/framework/triton/*.cc"
)

list(REMOVE_ITEM onnxruntime_training_srcs ${onnxruntime_training_framework_excluded_srcs})

onnxruntime_add_static_library(onnxruntime_training ${onnxruntime_training_srcs})
add_dependencies(onnxruntime_training onnx tensorboard ${onnxruntime_EXTERNAL_DEPENDENCIES})
onnxruntime_add_include_to_target(onnxruntime_training onnxruntime_common onnx onnx_proto tensorboard ${PROTOBUF_LIB} flatbuffers::flatbuffers re2::re2 Boost::mp11 safeint_interface)

# fix event_writer.cc 4100 warning
if(WIN32)
  target_compile_options(onnxruntime_training PRIVATE /wd4100)
endif()

target_include_directories(onnxruntime_training PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${eigen_INCLUDE_DIRS} PUBLIC ${onnxruntime_graph_header} ${MPI_CXX_INCLUDE_DIRS})

if(onnxruntime_USE_CUDA)
  target_include_directories(onnxruntime_training PRIVATE ${onnxruntime_CUDNN_HOME}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(onnxruntime_USE_NCCL)
  target_include_directories(onnxruntime_training PRIVATE ${NCCL_INCLUDE_DIRS})
endif()

if(onnxruntime_BUILD_UNIT_TESTS)
  set_target_properties(onnxruntime_training PROPERTIES FOLDER "ONNXRuntime")
  source_group(TREE ${ORTTRAINING_ROOT} FILES ${onnxruntime_training_srcs})

  set(ONNXRUNTIME_LIBS
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_MKLDNN}
    ${PROVIDERS_DML}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    onnxruntime_framework
  )

  if(onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    list(APPEND ONNXRUNTIME_LIBS Python::Python)
  endif()

  list(APPEND ONNXRUNTIME_LIBS
    onnxruntime_graph
    ${ONNXRUNTIME_MLAS_LIBS}
    onnxruntime_common
    onnxruntime_flatbuffers
    Boost::mp11 safeint_interface
  )

  if(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
    list(APPEND ONNXRUNTIME_LIBS onnxruntime_language_interop onnxruntime_pyop)
  endif()
endif()
