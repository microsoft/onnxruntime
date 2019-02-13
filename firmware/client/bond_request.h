#pragma once
#include "client/api.h"
#include <stddef.h>
#include <stdint.h>
#include "bond_struct.h"
#include "Parameters_types.c.h"
#ifdef __cplusplus
extern "C" {
#endif
namespace onnxruntime {
CLIENT_API int32_t BrainSlice_Request(const BrainSlice_Parameters* sku,
                           const bond_util::BondStruct* args,
                           uint32_t function_id,
                           size_t payloadSize,
                           void** payload,
                           void* message,
                           size_t* messageSize);
}

#ifdef __cplusplus
}
#endif
