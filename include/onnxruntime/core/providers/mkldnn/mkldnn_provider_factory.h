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
ORT_API_STATUS(OrtCreateMkldnnExecutionProviderFactory, int use_arena, _Out_ OrtProviderFactory** out);

#ifdef __cplusplus
}
#endif
