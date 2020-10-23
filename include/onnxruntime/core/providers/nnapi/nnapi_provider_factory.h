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
enum NNAPIFlag : unsigned char {
  NNAPI_FLAG_USE_FP16 = 0,
  NNAPI_FLAG_USE_NCHW = 1,
  NNAPI_FLAG_MAX,  // Keep NNAPI_FLAG_MAX at the end of the enum definition
};

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Nnapi, _In_ OrtSessionOptions* options, unsigned long nnapi_flags = 0);

#ifdef __cplusplus
}
#endif
