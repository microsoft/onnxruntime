# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

project(orttraining_external_custom_ops)
find_package(pybind11 REQUIRED)
find_package(PythonInterp 3)
find_package(PythonLibs 3)
add_compile_definitions(ONNX_ML=1)
add_compile_definitions(ONNX_NAMESPACE=onnx)
add_compile_definitions(ONNX_USER_LITE_PROTO=ON)
add_library(orttraining_external_custom_ops SHARED src/foo_op.cpp)

if ("${ONNXRUNTIME_EXTERNAL_INCLUDE}" STREQUAL "")
  execute_process (COMMAND python -c "import os; import onnxruntime; print(
                              os.path.join(os.path.join(os.path.dirname(onnxruntime.__file__), 'external'), 'include'))"
                   OUTPUT_VARIABLE ONNXRUNTIME_EXTERNAL_INCLUDE
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if ("${ONNX_INCLUDE}" STREQUAL "")
  execute_process (COMMAND python -c "import os; import onnx; print(os.path.dirname(os.path.dirname(onnx.__file__)))"
                   OUTPUT_VARIABLE ONNX_INCLUDE
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
target_include_directories(orttraining_external_custom_ops PUBLIC ${ONNX_INCLUDE} ${ONNXRUNTIME_EXTERNAL_INCLUDE})
pybind11_add_module(orttraining_external_custom_ops src/foo_op.cpp)
