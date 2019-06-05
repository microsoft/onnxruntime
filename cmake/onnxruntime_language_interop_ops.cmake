# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(NumPy)

if(NOT PYTHON_INCLUDE_DIR)
  set(PYTHON_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\""
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    RETURN_VALUE PYTHON_NOT_FOUND)
  if(${PYTHON_NOT_FOUND})
    message(FATAL_ERROR "Cannot get Python include directory. Is distutils installed?")
  endif(${PYTHON_NOT_FOUND})
endif(NOT PYTHON_INCLUDE_DIR)

if(NOT NUMPY_INCLUDE_DIR)
  set(NUMPY_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import numpy; print(numpy.get_include())\""
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)
  if(${NUMPY_NOT_FOUND})
    message(FATAL_ERROR "Cannot get NumPy include directory: Is NumPy installed?")
  endif(${NUMPY_NOT_FOUND})
endif(NOT NUMPY_INCLUDE_DIR)

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${NUMPY_INCLUDE_DIR})

file(GLOB onnxruntime_pyop_srcs "${ONNXRUNTIME_ROOT}/core/language_interop_ops/pyop.cc")
add_library(onnxruntime_pyop SHARED ${onnxruntime_pyop_srcs})
if (WIN32)
  set_target_properties(onnxruntime_pyop PROPERTIES LINK_FLAGS "/ignore:4199")
endif()
target_link_libraries(onnxruntime_pyop PUBLIC ${PYTHON_LIBRARIES})
