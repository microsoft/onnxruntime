// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param device_id hip device id, starts from zero.
 */
// declared in include/onnxruntime/core/session/onnxruntime_c_api.h for convenience and so we can provide a graceful
// error message if not enabled.
//ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_ROCM, _In_ OrtSessionOptions* options, int device_id, size_t gpu_mem_limit);

#ifdef __cplusplus
}
#endif
