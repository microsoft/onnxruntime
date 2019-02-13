#pragma once
#include "client/api.h"
#include <stddef.h>
#include <stdint.h>
#include "Parameters_types.c.h"
#ifdef __cplusplus
extern "C" {
#endif
namespace onnxruntime {
CLIENT_API int32_t BrainSlice_Response(const BrainSlice_Parameters* sku,
                           const void* message,
                           const size_t messageSize,
                           const void** payload,
                           size_t* payloadSize);
}

#ifdef __cplusplus
}
#endif
