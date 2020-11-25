#pragma once
#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);


#ifdef __cplusplus
}
#endif
