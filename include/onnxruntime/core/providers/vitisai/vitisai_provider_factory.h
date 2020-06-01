// Copyright (c) Xilinx Inc.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param use_arena zero: false. non-zero: true.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_VITISAI, _In_ OrtSessionOptions* options, const char *backend_type, int device_id);

#ifdef __cplusplus
}
#endif
