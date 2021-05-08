# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
file(GLOB_RECURSE onnxruntime_language_interop_torch_srcs
    "${ONNXRUNTIME_ROOT}/core/language_interop_ops/torch/*.h"
    "${ONNXRUNTIME_ROOT}/core/language_interop_ops/torch/*.cc"
)

onnxruntime_add_static_library(onnxruntime_language_interop_torch ${onnxruntime_language_interop_torch_srcs})
add_dependencies(onnxruntime_language_interop_torch onnxruntime_graph onnxruntime_util)
onnxruntime_add_include_to_target(onnxruntime_language_interop_torch onnxruntime_common onnxruntime_util onnxruntime_graph onnxruntime_framework onnx onnx_proto protobuf::libprotobuf flatbuffers)
target_include_directories(onnxruntime_language_interop_torch PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
target_include_directories(onnxruntime_language_interop_torch PRIVATE ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})
target_link_libraries(onnxruntime_language_interop_torch PRIVATE onnxruntime_util ${PYTHON_LIBRARIES})

if (onnxruntime_ENABLE_TRAINING)
  # DLPack is a header-only dependency
  set(DLPACK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external/dlpack/include)
  target_include_directories(onnxruntime_language_interop_torch PRIVATE ${ORTTRAINING_ROOT} ${DLPACK_INCLUDE_DIR})
endif()
# Enable compiler warnings
if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function")
endif()
