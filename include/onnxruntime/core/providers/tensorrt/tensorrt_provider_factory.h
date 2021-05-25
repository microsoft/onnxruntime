// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id);

/**
 * Use this API to create the configuration of a TensorRT Execution Provider.
 * \param out - pointer to the pointer of TensorRT EP provider options instance.
 */
ORT_API_STATUS(OrtCreateTensorRTProviderOptions, _Outptr_ OrtTensorRTProviderOptions** out);

/**
  * Use this API to set appropriate configuration knobs of a TensorRT Execution Provider.
  * \tensorrt_provider_options - OrtTensorRTProviderOptions instance
  * \provider_options_keys - array of UTF-8 null-terminated string for provider options keys
  * \provider_options_values - array of UTF-8 null-terminated string for provider options values
  * \num_keys - number of keys
  */
ORT_API_STATUS(OrtUpdateTensorRTProviderOptions, _Inout_ OrtTensorRTProviderOptions* tensorrt_provider_options,
                _In_reads_(num_keys) const char* const* provider_options_keys,
                _In_reads_(num_keys) const char* const* provider_options_values,
                _In_ size_t num_keys);

/**
  * Get configuration of a TensorRT Execution Provider.
  * \param allocator - a ptr to an instance of OrtAllocator obtained with CreateAllocator() or GetAllocatorWithDefaultOptions()
  *                      the specified allocator will be used to allocate continuous buffers for output strings and lengths.
  * \param ptr - is a UTF-8 null terminated string allocated using 'allocator'. The caller is responsible for using the same allocator to free it.
  */
ORT_API_STATUS(OrtGetTensorRTProviderOptions, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);

/**
  * Use this API to release the instance of configuration of a TensorRT Execution Provider.
  */
ORT_API(void, OrtReleaseTensorRTProviderOptions, _Frees_ptr_opt_ OrtTensorRTProviderOptions*);

#ifdef __cplusplus
}
#endif
