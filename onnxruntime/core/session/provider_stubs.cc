// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"

#if defined(__APPLE__) || defined(ORT_MINIMAL_BUILD) || !defined(USE_ROCM)
static OrtStatus* CreateNotEnabledStatus(const std::string& ep) {
  return OrtApis::CreateStatus(ORT_FAIL, (ep + " execution provider is not enabled in this build. ").c_str());
}
#endif

// we need stubs for functions called from C# when building an iOS app using Xamarin.
// in that case a static ORT library is used and the symbol needs to exist but doesn't need to be publicly exported.
// TODO: Not sure if we need to purely limit to iOS builds, so limit to __APPLE__ for now
#ifdef __APPLE__

#ifdef __cplusplus
extern "C" {
#endif

#ifndef USE_DML
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_DML, _In_ OrtSessionOptions* options, int device_id) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(device_id);
  return CreateNotEnabledStatus("DML");
}
#endif

#ifndef USE_MIGRAPHX
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_MIGraphX,
                    _In_ OrtSessionOptions* options, int device_id) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(device_id);
  return CreateNotEnabledStatus("MIGraphX");
}
#endif

#ifndef USE_ROCM
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_ROCM,
                    _In_ OrtSessionOptions* options, int device_id, size_t gpu_mem_limit) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(device_id);
  ORT_UNUSED_PARAMETER(gpu_mem_limit);
  return CreateNotEnabledStatus("ROCM");
}
#endif

#ifndef USE_NNAPI
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nnapi,
                    _In_ OrtSessionOptions* options, uint32_t nnapi_flags) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(nnapi_flags);
  return CreateNotEnabledStatus("NNAPI");
}
#endif

#ifndef USE_NUPHAR
ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Nuphar,
                    _In_ OrtSessionOptions* options, int allow_unaligned_buffers, _In_ const char* settings) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(allow_unaligned_buffers);
  ORT_UNUSED_PARAMETER(settings);
  return CreateNotEnabledStatus("Nuphar");
}
#endif

#ifdef __cplusplus
}
#endif

#endif  // __APPLE__

/*
OrtApis::SessionOptionsAppendExecutionProvider_<EP> stubs for EPs not included in this build.

2 cases:

1) EP is included in ORT via build settings. Need to include stub if EP not enabled.
2) EP is built as separate library and uses the provider bridge. Need to include in a minimal build 
   as the provider bridge is excluded in that case.

TODO: When the NNAPI or CoreML EPs are setup to use the provider bridge the source code for that will be included
      in a minimal build and these stubs should move to provider_bridge_ort.cc.
*/

// EPs in the first case
#ifndef USE_ROCM
ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_ROCM,
                    _In_ OrtSessionOptions* options, _In_ const OrtROCMProviderOptions* rocm_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(rocm_options);
  return CreateNotEnabledStatus("ROCM");
}
#endif

// EPs in the second case
#if defined(ORT_MINIMAL_BUILD)
ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_CUDA,
                    _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::GetCurrentGpuDeviceId, _In_ int* device_id) {
  ORT_UNUSED_PARAMETER(device_id);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::SetCurrentGpuDeviceId, _In_ int device_id) {
  ORT_UNUSED_PARAMETER(device_id);
  return CreateNotEnabledStatus("CUDA");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(provider_options);
  return CreateNotEnabledStatus("OpenVINO");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT,
                    _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(tensorrt_options);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT_V2,
                    _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options) {
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(tensorrt_options);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorRTProviderOptions, _Outptr_ OrtTensorRTProviderOptionsV2** out) {
  ORT_UNUSED_PARAMETER(out);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::UpdateTensorRTProviderOptions,
                    _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys) {
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(provider_options_keys);
  ORT_UNUSED_PARAMETER(provider_options_values);
  ORT_UNUSED_PARAMETER(num_keys);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API_STATUS_IMPL(OrtApis::GetTensorRTProviderOptionsAsString,
                    _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** ptr) {
  ORT_UNUSED_PARAMETER(tensorrt_options);
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(ptr);
  return CreateNotEnabledStatus("TensorRT");
}

ORT_API(void, OrtApis::ReleaseTensorRTProviderOptions, _Frees_ptr_opt_ OrtTensorRTProviderOptionsV2* ptr) {
  ORT_UNUSED_PARAMETER(ptr);
}

#endif
