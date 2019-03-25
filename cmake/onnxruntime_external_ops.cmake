# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(PythonLibs)
file(GLOB onnxruntime_pyop_srcs "${ONNXRUNTIME_ROOT}/core/external_ops/pyop.cc")
add_library(onnxruntime_pyop SHARED ${onnxruntime_pyop_srcs} ${PYTHON_LIBRARIES})
onnxruntime_add_include_to_target(onnxruntime_pyop gsl)
add_dependencies(onnxruntime_pyop onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_pyop PUBLIC "${PROJECT_SOURCE_DIR}/include"
    ${PYTHON_INCLUDE_DIRS} include_directories("~/.local/lib/python3.6/site-packages/numpy/core/include/"))
target_link_libraries(onnxruntime_pyop PRIVATE onnxruntime onnx onnx_proto  protobuf::libprotobuf)

