// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Stvm, _In_ OrtSessionOptions* options, _In_ const char* settings);

#ifdef __cplusplus
}
#endif


