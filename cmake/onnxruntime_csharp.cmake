# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set (CSHARP_ROOT ${PROJECT_SOURCE_DIR}/../csharp)
set (CSHARP_MASTER_TARGET OnnxRuntime.CSharp)
set (CSHARP_MASTER_PROJECT ${CSHARP_ROOT}/OnnxRuntime.CSharp.proj )
include(CSharpUtilities)

include_external_msproject(${CSHARP_MASTER_TARGET}
                           ${CSHARP_MASTER_PROJECT}
                           onnxruntime   # make it depend on the native onnxruntime project
                           )

# generate Directory.Build.props
set(DIRECTORY_BUILD_PROPS_COMMENT "WARNING: This is a generated file, please do not check it in!")
configure_file(${CSHARP_ROOT}/Directory.Build.props.in
               ${CSHARP_ROOT}/Directory.Build.props
               @ONLY)
