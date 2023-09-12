// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Shl, _In_ OrtSessionOptions* options, _In_ const char* opt_str);

#ifdef __cplusplus
}
#endif
