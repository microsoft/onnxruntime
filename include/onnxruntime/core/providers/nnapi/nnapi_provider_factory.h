// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime_c_api.h"

// NNAPIFlags are bool options we want to set for NNAPI EP
// This enum is defined as bit flats, and cannot have negative value
// To generate a unsigned long nnapi_flags for using with OrtSessionOptionsAppendExecutionProvider_Nnapi below,
//   unsigned long nnapi_flags = 0;
//   nnapi_flags |= NNAPI_FLAG_USE_FP16;
enum NNAPIFlags {
  NNAPI_FLAG_USE_NONE = 0x000,

  // Using fp16 relaxation in NNAPI EP, this may improve perf but may also reduce precision
  NNAPI_FLAG_USE_FP16 = 0x001,

  // Use NCHW layout in NNAPI EP, this is only available after Android API level 29
  // Please note for now, NNAPI perform worse using NCHW compare to using NHWC
  NNAPI_FLAG_USE_NCHW = 0x002,

  // Keep NNAPI_FLAG_MAX at the end of the enum definition
  // And assign the last NNAPIFlag to it
  NNAPI_FLAG_LAST = NNAPI_FLAG_USE_NCHW,
};

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Nnapi, _In_ OrtSessionOptions* options, unsigned long nnapi_flags);

#ifdef __cplusplus
}
#endif
