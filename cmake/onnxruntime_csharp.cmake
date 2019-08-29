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
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_CUDA,")
endif()

if (onnxruntime_USE_MKLDNN)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_MKLDNN,")
endif()

if (onnxruntime_USE_TENSORRT)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_TENSORRT,")
endif()

if (onnxruntime_USE_OPENVINO)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_OPENVINO,")
endif()

if (onnxruntime_USE_NGRAPH)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_NGRAPH,")
endif()

include(CSharpUtilities)

include_external_msproject(${CSHARP_MASTER_TARGET}
                           ${CSHARP_MASTER_PROJECT}
                           ${CSHARP_DEPENDS}
                           )

# generate Directory.Build.props
set(DIRECTORY_BUILD_PROPS_COMMENT "WARNING: This is a generated file, please do not check it in!")
configure_file(${CSHARP_ROOT}/Directory.Build.props.in
               ${CSHARP_ROOT}/Directory.Build.props
               @ONLY)
