// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

/**
 * \param device_type openvino device type and precision. Could be any of
 * CPU_FP32, GPU_FP32, GPU_FP16, MYRIAD_FP16, VAD-M_FP16 or VAD-F_FP32.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_OpenVINO,
    _In_ OrtSessionOptions* options, _In_ const char* device_type);

/**
 * \param settings_str string of Key-Value pairs with '\n' used to delimit
 * pairs and '|' used to delimit key and value within a pair.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProviderEx_OpenVINO,
    _In_ OrtSessionOptions* options, _In_ const char* settings_str);

#ifdef __cplusplus
}
#endif
