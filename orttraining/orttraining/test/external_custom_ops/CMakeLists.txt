cmake_minimum_required(VERSION 3.20)
project(orttraining_external_custom_ops)
find_package(pybind11 REQUIRED)
find_package(PythonInterp 3)
find_package(PythonLibs 3)
add_compile_definitions(ONNX_ML=1)
add_compile_definitions(ONNX_NAMESPACE=onnx)
add_compile_definitions(ONNX_USER_LITE_PROTO=ON)
add_library(orttraining_external_custom_ops SHARED src/foo_op.cpp)

if ("${PYTHON_SITE_PACKAGES}" STREQUAL "")
  execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
                    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if ("${ONNX_INCLUDE_DIR}" STREQUAL "")
  execute_process ( COMMAND python -c "import os;import onnx; print(os.path.dirname(os.path.dirname(onnx.__file__)))"
                    OUTPUT_VARIABLE ONNX_INCLUDE_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

target_include_directories(orttraining_external_custom_ops PUBLIC ${ONNX_INCLUDE_DIR} ${PYTHON_SITE_PACKAGES}/onnxruntime/external/include/)
pybind11_add_module(orttraining_external_custom_ops src/foo_op.cpp)
