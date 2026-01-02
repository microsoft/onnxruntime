#pragma once

#include <onnxruntime_c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Register ThresholdedRelu custom operator with ONNX Runtime
 * 
 * This function is the entry point for custom operator registration.
 * The function name MUST be exactly "RegisterCustomOps" for automatic discovery.
 * 
 * @param options Session options to register the operators with
 * @param api ONNX Runtime API base pointer
 * @return OrtStatus* Status of the registration operation
 */
#ifdef _WIN32
__declspec(dllexport)
#endif
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);

#ifdef __cplusplus
}
#endif