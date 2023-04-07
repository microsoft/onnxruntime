// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include "core/providers/shared_library/provider_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_MyEP, _In_ OrtSessionOptions* options, int device_id);

ORT_API(onnxruntime::Provider*, GetProvider);

ORT_API(size_t, ProviderHashFunc, const void* options);

#ifdef __cplusplus
}
#endif