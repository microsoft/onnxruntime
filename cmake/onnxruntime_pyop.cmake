# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_pywrapper_srcs "${ONNXRUNTIME_ROOT}/core/language_interop_ops/pyop/pywrapper.cc")
add_library(onnxruntime_pywrapper SHARED ${onnxruntime_pywrapper_srcs})
if (WIN32)
  set_target_properties(onnxruntime_pywrapper PROPERTIES LINK_FLAGS "/ignore:4199")
endif()
target_include_directories(onnxruntime_pywrapper PRIVATE ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})
target_link_libraries(onnxruntime_pywrapper PRIVATE ${PYTHON_LIBRARIES})

file(GLOB onnxruntime_pyop_srcs "${ONNXRUNTIME_ROOT}/core/language_interop_ops/pyop/pyop.cc")
add_library(onnxruntime_pyop ${onnxruntime_pyop_srcs})
add_dependencies(onnxruntime_pyop onnxruntime_graph)
onnxruntime_add_include_to_target(onnxruntime_pyop onnxruntime_common onnxruntime_graph onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
target_include_directories(onnxruntime_pyop PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})

