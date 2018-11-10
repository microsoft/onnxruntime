# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set (CSHARP_ROOT ${PROJECT_SOURCE_DIR}/../csharp)
set (CSHARP_MASTER_TARGET Microsoft.ML.OnnxRuntime)
set (CSHARP_MASTER_PROJECT "${CSHARP_ROOT}/OnnxRuntime/OnnxRuntime.csproj" )
include(CSharpUtilities)

include_external_msproject(${CSHARP_MASTER_TARGET}
                           "${CSHARP_MASTER_PROJECT}"
                           onnxruntime   # make it depend on the native onnxruntime project
                           )
