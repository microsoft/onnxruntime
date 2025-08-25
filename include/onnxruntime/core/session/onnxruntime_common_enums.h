// Shared public enums used by both onnxruntime_c_api.h and onnxruntime_ep_c_api.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Compatibility state of a compiled model relative to an execution provider.
typedef enum OrtCompiledModelCompatibility {
    OrtCompiledModelCompatibility_EP_NOT_APPLICABLE = 0,
    OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL,
    OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION,
    OrtCompiledModelCompatibility_EP_UNSUPPORTED,
  } OrtCompiledModelCompatibility;

#ifdef __cplusplus
}
#endif
