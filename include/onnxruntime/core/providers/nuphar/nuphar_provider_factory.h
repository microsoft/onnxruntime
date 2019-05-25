// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * \param device_id nuphar device id, starts from zero.
 * \param target_str TVM target string.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Nuphar, _In_ OrtSessionOptions* options, int allow_unaligned_buffers, int device_id, _In_ const char* target_str);

#ifdef __cplusplus
}
#endif
