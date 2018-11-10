// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/framework/error_code.h"
//This file is part of the public C API
#ifdef __cplusplus
extern "C" {
#endif
typedef enum ONNXRuntimeAllocatorType {
  ONNXRuntimeDeviceAllocator = 0,
  ONNXRuntimeArenaAllocator = 1
} ONNXRuntimeAllocatorType;

/**
   memory types for allocator, exec provider specific types should be extended in each provider
*/
typedef enum ONNXRuntimeMemType {
  ONNXRuntimeMemTypeCPUInput = -2,                      // Any CPU memory used by non-CPU execution provider
  ONNXRuntimeMemTypeCPUOutput = -1,                     // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
  ONNXRuntimeMemTypeCPU = ONNXRuntimeMemTypeCPUOutput,  // temporary CPU accessible memory allocated by non-CPU execution provider, i.e. CUDA_PINNED
  ONNXRuntimeMemTypeDefault = 0,                        // the default allocator for execution provider
} ONNXRuntimeMemType;

DEFINE_RUNTIME_CLASS(ONNXRuntimeAllocatorInfo);

ONNXRUNTIME_API_STATUS(ONNXRuntimeCreateAllocatorInfo, _In_ const char* name1, enum ONNXRuntimeAllocatorType type, int id1, enum ONNXRuntimeMemType mem_type1, _Out_ ONNXRuntimeAllocatorInfo** out);

/**
 * Test if two allocation info are equal
 * \return 0, equal. zero, not equal
 */
ONNXRUNTIME_API(int, ONNXRuntimeCompareAllocatorInfo, _In_ ONNXRuntimeAllocatorInfo* info1, _In_ ONNXRuntimeAllocatorInfo* info2)
ONNXRUNTIME_ALL_ARGS_NONNULL;
/**
 * Do not free the returned value
 */
ONNXRUNTIME_API(const char*, ONNXRuntimeAllocatorInfoGetName, _In_ ONNXRuntimeAllocatorInfo* ptr);
ONNXRUNTIME_API(int, ONNXRuntimeAllocatorInfoGetId, _In_ ONNXRuntimeAllocatorInfo* ptr);
ONNXRUNTIME_API(ONNXRuntimeMemType, ONNXRuntimeAllocatorInfoGetMemType, _In_ ONNXRuntimeAllocatorInfo* ptr);
ONNXRUNTIME_API(ONNXRuntimeAllocatorType, ONNXRuntimeAllocatorInfoGetType, _In_ ONNXRuntimeAllocatorInfo* ptr);
#ifdef __cplusplus
}
#endif