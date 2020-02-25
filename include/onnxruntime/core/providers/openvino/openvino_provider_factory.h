// Copyright(C) 2020 Intel Corporation
// Licensed under the MIT License

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param device_id cuda device id, starts from zero.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_OpenVINO,
    _In_ OrtSessionOptions* options, const char* device_id);

#ifdef __cplusplus
}
#endif
