// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Xnnpack,
                          _In_ OrtSessionOptions* options);

#ifdef __cplusplus
}
#endif
