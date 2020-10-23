// Copyright 2019 JD.com Inc. JD AI
#pragma once

#include "onnxruntime_c_api.h"

// NNAPIFlag are bool options we want to set for NNAPI EP
// To generate a unsigned long nnapi_flags for using with OrtSessionOptionsAppendExecutionProvider_Nnapi below,
// If using C
//   unsigned long nnapi_flags = 0;
//   nnapi_flags |= 1UL << NNAPI_FLAG_USE_FP16;
// If using C++
//   std::bitset<NNAPI_FLAG_MAX> _nnapi_flags;
//   _nnapi_flags.set(NNAPI_FLAG_USE_FP16);
//   unsigned long nnapi_flags = _nnapi_flags.to_ulong();
enum NNAPIFlag {
  // Using fp16 relaxation in NNAPI EP, this may improve perf but may also reduce precision
  NNAPI_FLAG_USE_FP16 = 0,

  // Use NCHW layout in NNAPI EP, this is only available after Android API level 29
  // Please note for now, NNAPI perform worse using NCHW compare to using NHWC
  NNAPI_FLAG_USE_NCHW = 1,

  // Keep NNAPI_FLAG_MAX at the end of the enum definition
  NNAPI_FLAG_MAX,
};

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Nnapi, _In_ OrtSessionOptions* options, unsigned long nnapi_flags);

#ifdef __cplusplus
}
#endif
