// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param use_arena zero: false. non-zero: true.
 * \param out Call OrtReleaseObject() method when you no longer need to use it.
 */
ORT_API_STATUS(OrtCreateCpuExecutionProviderFactory, int use_arena, _Out_ OrtProviderFactoryInterface*** out)
ORT_ALL_ARGS_NONNULL;

ORT_API_STATUS(OrtCreateCpuAllocatorInfo, enum OrtAllocatorType type, enum OrtMemType mem_type1, _Out_ OrtAllocatorInfo** out)
ORT_ALL_ARGS_NONNULL;

#ifdef __cplusplus
}
#endif
