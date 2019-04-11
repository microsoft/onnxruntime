# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("~/.local/lib/python3.6/site-packages/numpy/core/include/")
#message("PYTHON LIBS: " ${PYTHON_LIBRARIES})
#message("PYTHON INCS: " ${PYTHON_INCLUDE_DIRS})
file(GLOB onnxruntime_pyop_srcs "${ONNXRUNTIME_ROOT}/core/external_ops/pyop.cc")
#add_executable(onnxruntime_pyop ${onnxruntime_pyop_srcs})
add_library(onnxruntime_pyop SHARED ${onnxruntime_pyop_srcs})
#add_executable(onnxruntime_pyop ${onnxruntime_pyop_srcs})
#onnxruntime_add_include_to_target(onnxruntime_pyop gsl)
#add_dependencies(onnxruntime_pyop onnxruntime_common onnxruntime_providers onnxruntime_optimizer onnxruntime_framework onnxruntime_graph onnxruntime_session onnxruntime_util onnx onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
#target_include_directories(onnxruntime_pyop PUBLIC "${PROJECT_SOURCE_DIR}/include")
#target_link_libraries(onnxruntime_pyop PUBLIC onnxruntime_session onnxruntime_optimizer onnxruntime_providers re2 onnxruntime_mlas onnxruntime_util onnxruntime_framework onnxruntime_graph onnxruntime_common onnx dl onnx_proto protobuf::libprotobuf libprotobuf ${PYTHON_LIBRARIES})
target_link_libraries(onnxruntime_pyop PUBLIC ${PYTHON_LIBRARIES})


