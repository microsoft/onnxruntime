#pragma once
#include "client/api.h"
#include <stddef.h>
#include <stdint.h>
#include "Parameters_types.c.h"
#ifdef __cplusplus
extern "C" {
#endif
namespace BrainSlice {
CLIENT_API int32_t GetParametersRequest(void* message, size_t* messageSize);
CLIENT_API int32_t GetParametersResponse(const void* message, const size_t messageSize, BrainSlice_Parameters* parameters);
}
#ifdef __cplusplus
}
#endif
