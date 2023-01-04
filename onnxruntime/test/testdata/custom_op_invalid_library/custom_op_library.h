// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <onnxruntime_c_api.h>

#ifdef _WIN32
#define OV_WRAPPER_EXPORT __declspec(dllexport)
#elif __APPLE__
#define OV_WRAPPER_EXPORT __attribute__((visibility("default")))
#else
#define OV_WRAPPER_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

OV_WRAPPER_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base);

#ifdef __cplusplus
}
#endif
