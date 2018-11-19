// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/error_code.h"
#include "core/framework/allocator_info.h"
#include "core/framework/onnx_object.h"

#ifdef __cplusplus
extern "C" {
#endif
//inherented from ONNXObject
typedef struct ONNXRuntimeAllocatorInteface {
  struct ONNXObject parent;
  void*(ONNXRUNTIME_API_STATUSCALL* Alloc)(void* this_, size_t size);
  void(ONNXRUNTIME_API_STATUSCALL* Free)(void* this_, void* p);
  const struct ONNXRuntimeAllocatorInfo*(ONNXRUNTIME_API_STATUSCALL* Info)(const void* this_);
} ONNXRuntimeAllocatorInteface;

typedef ONNXRuntimeAllocatorInteface* ONNXRuntimeAllocator;

ONNXRUNTIME_API(void*, ONNXRuntimeAllocatorAlloc, _Inout_ ONNXRuntimeAllocator* ptr, size_t size);
ONNXRUNTIME_API(void, ONNXRuntimeAllocatorFree, _Inout_ ONNXRuntimeAllocator* ptr, void* p);
ONNXRUNTIME_API(const struct ONNXRuntimeAllocatorInfo*, ONNXRuntimeAllocatorGetInfo, _In_ const ONNXRuntimeAllocator* ptr);

#ifdef __cplusplus
}
#endif