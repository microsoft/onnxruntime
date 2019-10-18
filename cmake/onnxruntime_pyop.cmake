# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(PYTHON_NOT_FOUND false)
exec_program("${PYTHON_EXECUTABLE}" # get python include path
  ARGS "-c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\"" 
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
  RETURN_VALUE PYTHON_NOT_FOUND)
if(${PYTHON_NOT_FOUND})
  message(FATAL_ERROR "Cannot get Python include directory. Is distutils installed?")
else()
  exec_program("${PYTHON_EXECUTABLE}" # get python library path
    ARGS "-c \"
import os
import re
import sys
major = sys.version_info.major
minor = sys.version_info.minor
regex = re.compile('^(libpython|python)'+str(major)+'\.?'+str(minor)+'m?\.(so|lib|dylib)$')
pydir = os.path.abspath(os.path.join(os.path.dirname(sys.executable),'..'))
for r,d,fs in os.walk(pydir):
  for f in fs:
    if regex.match(f):
      print (os.path.join(r,f))
      sys.exit()
\"
"
    OUTPUT_VARIABLE PYTHON_LIBRARIES)
endif(${PYTHON_NOT_FOUND})

exec_program("${PYTHON_EXECUTABLE}" # get numpy include path
  ARGS "-c \"import numpy; print(numpy.get_include())\""
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  RETURN_VALUE NUMPY_NOT_FOUND)
if(${NUMPY_NOT_FOUND})
  message(FATAL_ERROR "Cannot get NumPy include directory: Is NumPy installed?")
endif(${NUMPY_NOT_FOUND})

message("PYTHON EXE: " ${PYTHON_EXECUTABLE})
message("PYTHON INC: " ${PYTHON_INCLUDE_DIR})
message("PYTHON LIB: " ${PYTHON_LIBRARIES})
message("NUMPY  INC: " ${NUMPY_INCLUDE_DIR})

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${NUMPY_INCLUDE_DIR})

file(GLOB onnxruntime_pywrapper_srcs "${ONNXRUNTIME_ROOT}/core/language_interop_ops/pyop/pywrapper.cc")
add_library(onnxruntime_pywrapper SHARED ${onnxruntime_pywrapper_srcs})
if (WIN32)
  set_target_properties(onnxruntime_pywrapper PROPERTIES LINK_FLAGS "/ignore:4199")
endif()
target_link_libraries(onnxruntime_pywrapper PUBLIC ${PYTHON_LIBRARIES})

file(GLOB onnxruntime_pyop_srcs "${ONNXRUNTIME_ROOT}/core/language_interop_ops/pyop/pyop.cc")
add_library(onnxruntime_pyop ${onnxruntime_pyop_srcs})
add_dependencies(onnxruntime_pyop onnxruntime_graph)
onnxruntime_add_include_to_target(onnxruntime_pyop onnxruntime_common onnxruntime_graph onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
target_include_directories(onnxruntime_pyop PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})

