// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param use_fp16 zero: false. non-zero: true.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_OpenCL, _In_ OrtSessionOptions* options, int use_fp16, int enable_auto_tune)
ORT_ALL_ARGS_NONNULL;

#ifdef __cplusplus
}
#endif
