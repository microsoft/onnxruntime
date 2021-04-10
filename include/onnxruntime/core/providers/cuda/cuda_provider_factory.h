// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param device_id cuda device id, starts from zero.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id);
ORT_API_STATUS(SessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* cuda_options);

#ifdef __cplusplus
}
#endif
