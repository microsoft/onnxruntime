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

if (onnxruntime_USE_NUPHAR)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_NUPHAR,")
endif()

include(CSharpUtilities)

include_external_msproject(Microsoft.ML.OnnxRuntime
                           ${CSHARP_ROOT}/src/Microsoft.ML.OnnxRuntime/Microsoft.ML.OnnxRuntime.csproj
                           ${CSHARP_DEPENDS}
                           )

include_external_msproject(Microsoft.ML.OnnxRuntime.InferenceSample
                           ${CSHARP_ROOT}/sample/Microsoft.ML.OnnxRuntime.InferenceSample/Microsoft.ML.OnnxRuntime.InferenceSample.csproj
                           ${CSHARP_DEPENDS}
                           )
include_external_msproject(Microsoft.ML.OnnxRuntime.Tests
                           ${CSHARP_ROOT}/test/Microsoft.ML.OnnxRuntime.Tests/Microsoft.ML.OnnxRuntime.Tests.csproj
                           ${CSHARP_DEPENDS}
                           )
include_external_msproject(Microsoft.ML.OnnxRuntime.PerfTool
                           ${CSHARP_ROOT}/tools/Microsoft.ML.OnnxRuntime.PerfTool/Microsoft.ML.OnnxRuntime.PerfTool.csproj
                           ${CSHARP_DEPENDS}
                           )

#Exclude them from the ALL_BUILD target, otherwise it will trigger errors like:
#"Error : Project 'cmake\..\csharp\src\Microsoft.ML.OnnxRuntime\Microsoft.ML.OnnxRuntime.csproj' targets 'netstandard1.1'. It cannot be referenced by a project that targets '.NETFramework,Version=v4.0'."
#We can't fix it because cmake only supports the "TargetFrameworkVersion" property, not "TargetFramework".
set_target_properties(Microsoft.ML.OnnxRuntime Microsoft.ML.OnnxRuntime.InferenceSample Microsoft.ML.OnnxRuntime.Tests Microsoft.ML.OnnxRuntime.PerfTool PROPERTIES EXCLUDE_FROM_ALL 1)

# generate Directory.Build.props
set(DIRECTORY_BUILD_PROPS_COMMENT "WARNING: This is a generated file, please do not check it in!")
configure_file(${CSHARP_ROOT}/Directory.Build.props.in
               ${CSHARP_ROOT}/Directory.Build.props
               @ONLY)
