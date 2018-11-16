// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/error_code.h"
#ifdef __cplusplus
extern "C" {
#endif
struct ONNXRuntimeRunOptions;
typedef struct ONNXRuntimeRunOptions ONNXRuntimeRunOptions;
typedef ONNXRuntimeRunOptions* ONNXRuntimeRunOptionsPtr;
/**
 * \return A pointer of the newly created object. The pointer should be freed by ONNXRuntimeReleaseObject after use
 */
ONNXRUNTIME_API(ONNXRuntimeRunOptions*, ONNXRuntimeCreateRunOptions);

ONNXRUNTIME_API_STATUS(ONNXRuntimeRunOptionsSetRunLogVerbosityLevel, _In_ ONNXRuntimeRunOptions*, unsigned int);
ONNXRUNTIME_API_STATUS(ONNXRuntimeRunOptionsSetRunTag, _In_ ONNXRuntimeRunOptions*, _In_ const char* run_tag);

ONNXRUNTIME_API(unsigned int, ONNXRuntimeRunOptionsGetRunLogVerbosityLevel, _In_ ONNXRuntimeRunOptions*);
ONNXRUNTIME_API(const char*, ONNXRuntimeRunOptionsGetRunTag, _In_ ONNXRuntimeRunOptions*);

// set a flag so that any running ONNXRuntimeRunInference* calls that are using this instance of ONNXRuntimeRunOptions
// will exit as soon as possible if the flag is true.
ONNXRUNTIME_API(void, ONNXRuntimeRunOptionsSetTerminate, _In_ ONNXRuntimeRunOptions*, _In_ bool value);

#ifdef __cplusplus
}
#endif
