// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param use_arena zero: false. non-zero: true.
 * \param out Call ONNXRuntimeReleaseObject() method when you no longer need to use it.
 */
ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateMkldnnExecutionProviderFactory, int use_arena, _Out_ ONNXRuntimeProviderFactoryPtr** out);

#ifdef __cplusplus
}
#endif