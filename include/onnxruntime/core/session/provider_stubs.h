// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

/*
Stubs for the function that adds the EP to the session options for EPs that are not supported in this build. 

There are two ways this addition to SessionOptions currently happens:

1) Via a function that is exported directly from the EP code (usually from the provider factory). 
   The naming convention is OrtSessionOptionsAppendExecutionProvider_<EP>

2) Via an entry in the OrtApis struct.
   The naming convention is SessionOptionsAppendExecutionProvider_<EP>

When an EP is an external library the provider bridge will provide the entry point.

NOTE: The function of style #1 is easier to use from places like the C# and Java bindings as it does not use
a struct for passing the options.

When adding a new EP that is not behind the provider bridge you need to:
  1) Add the ORT_API_STATUS entry for the function if EP not enabled in build to provider_stubs.h
  2) Add the ORT_API_STATUS_IMPL implementation if EP not enabled in build to provider_stubs.cc
  3) Add the exported symbol name for the stub to the symbols.txt file in the onnxruntime/core/session directory.
*/

// #1 entry points
#ifndef USE_DML
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_DML, _In_ OrtSessionOptions* options, int device_id);
#endif

#ifndef USE_MIGRAPHX
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_MIGraphX, _In_ OrtSessionOptions* options, int device_id);
#endif

#ifndef USE_ROCM
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_ROCM,
               _In_ OrtSessionOptions* options, int device_id, size_t gpu_mem_limit);
#endif

#ifndef USE_NNAPI
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Nnapi, _In_ OrtSessionOptions* options, uint32_t nnapi_flags);
#endif

#ifndef USE_NUPHAR
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Nuphar,
               _In_ OrtSessionOptions* options, int allow_unaligned_buffers, _In_ const char* settings);
#endif

// #2 entry points
#ifndef USE_CUDA
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id);
#endif

#ifndef USE_DNNL
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Dnnl, _In_ OrtSessionOptions* options, int use_arena);
#endif

#ifndef USE_OPENVINO
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_OpenVINO, _In_ OrtSessionOptions* options,
               _In_ const char* device_type);
#endif

#ifndef USE_TENSORRT
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id);
#endif
