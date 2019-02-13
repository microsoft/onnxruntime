// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/session/onnxruntime_c_api.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_BrainSlice, _In_ OrtSessionOptions* options, uint32_t ip, bool load_firmware, _In_ const char* instr_path, _In_ const char* data_path, _In_ const char* schema_path);

#ifdef __cplusplus
}
#endif
