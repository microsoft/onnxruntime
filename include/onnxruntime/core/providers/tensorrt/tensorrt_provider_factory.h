// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include "core/session/ort_apis.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id);
ORT_API_STATUS(OrtApis::SessionOptionsAppendExecutionProvider_TensorRT, _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options);

#ifdef __cplusplus
}
#endif
