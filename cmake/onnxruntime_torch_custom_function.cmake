# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
message("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
message("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
message("Build library for using Pytorch's custom autograd.Function")
message("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
message("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
file(GLOB onnxruntime_torch_custom_function_srcs "${ONNXRUNTIME_ROOT}/core/torch_custom_function/*.cc")
add_library(onnxruntime_torch_custom_function ${onnxruntime_torch_custom_function_srcs})
target_compile_options(onnxruntime_torch_custom_function PRIVATE -fvisibility=hidden)
target_include_directories(onnxruntime_torch_custom_function PRIVATE ${PYTHON_INCLUDE_DIR} ${pybind11_INCLUDE_DIRS})
target_link_libraries(onnxruntime_torch_custom_function PRIVATE ${PYTHON_LIBRARIES} ${pybind11_lib})