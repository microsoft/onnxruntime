// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param use_arena zero: false. non-zero: true.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_VITISAI, _In_ OrtSessionOptions* options, _In_ const char* opt_str);

#ifdef __cplusplus
}
#endif
