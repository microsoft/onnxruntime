#pragma once
#include "client/api.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
namespace BrainSlice {
CLIENT_API int32_t LoadFirmwareAPI(
    uint32_t* instructions,
    size_t instruction_size,
    uint32_t* data,
    size_t data_size,
    uint64_t* schema,
    size_t schema_size,
    void* message, size_t* messageSize);
}

#ifdef __cplusplus
}
#endif
