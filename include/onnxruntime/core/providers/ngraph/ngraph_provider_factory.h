// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param use_arena zero: false. non-zero: true.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_NGraph, _In_ OrtSessionOptions* options, _In_ const char* ng_backend_type);

#ifdef __cplusplus
}
#endif
