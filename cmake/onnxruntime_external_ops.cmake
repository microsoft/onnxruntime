# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(PythonLibs)
if (WIN32)
  execute_process(COMMAND python -c "import numpy;print(numpy.__file__[:-11]+'core\\include')" OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)
else()
  execute_process(COMMAND python -c "import numpy;print(numpy.__file__[:-11]+'core/include')" OUTPUT_VARIABLE NUMPY_INCLUDE_DIR)
endif()
message("PYTHON_LIBS: " ${PYTHON_LIBRARIES})
message("NUMPY_INCLUDE_DIR: " ${NUMPY_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("${NUMPY_INCLUDE_DIR}")
file(GLOB onnxruntime_pyop_srcs "${ONNXRUNTIME_ROOT}/core/external_ops/pyop.cc")
add_library(onnxruntime_pyop SHARED ${onnxruntime_pyop_srcs})
target_link_libraries(onnxruntime_pyop PUBLIC ${PYTHON_LIBRARIES})