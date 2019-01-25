// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * \param device_id trt device id, starts from zero.
 * \param out Call OrtReleaseObject() method when you no longer need to use it.
 */
ORT_API_STATUS(OrtCreateTRTExecutionProviderFactory, int device_id, _Out_ OrtProviderFactoryInterface*** out);

#ifdef __cplusplus
}
#endif

