// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <onnxruntime_c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base);

#ifdef __cplusplus
}
#endif
