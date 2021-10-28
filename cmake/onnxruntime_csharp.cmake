# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set (CSHARP_ROOT ${PROJECT_SOURCE_DIR}/../csharp)
set (CSHARP_MASTER_TARGET OnnxRuntime.CSharp)
set (CSHARP_MASTER_PROJECT ${CSHARP_ROOT}/OnnxRuntime.CSharp.proj)
if (onnxruntime_RUN_ONNX_TESTS)
  set (CSHARP_DEPENDS onnxruntime ${test_data_target})
else()
  set (CSHARP_DEPENDS onnxruntime)
endif()

if (onnxruntime_USE_CUDA)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_CUDA;")
endif()

if (onnxruntime_USE_DNNL)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_DNNL;")
endif()

if (onnxruntime_USE_DML)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_DML;")
endif()

if (onnxruntime_USE_MIGRAPHX)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_MIGRAPHX;")
endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_NNAPI;")
endif()

if (onnxruntime_USE_NUPHAR)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_NUPHAR;")
endif()

if (onnxruntime_USE_OPENVINO)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_OPENVINO;")
endif()

if (onnxruntime_USE_ROCM)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_ROCM;")
endif()

if (onnxruntime_USE_TENSORRT)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_TENSORRT;")
endif()

include(CSharpUtilities)

# generate Directory.Build.props
set(DIRECTORY_BUILD_PROPS_COMMENT "WARNING: This is a generated file, please do not check it in!")
configure_file(${CSHARP_ROOT}/Directory.Build.props.in
               ${CSHARP_ROOT}/Directory.Build.props
               @ONLY)
