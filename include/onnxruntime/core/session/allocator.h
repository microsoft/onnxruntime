// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/error_code.h"
#include "core/framework/allocator_info.h"
#include "core/framework/onnx_object.h"

//inherented from ONNXObject
typedef struct ONNXRuntimeAllocatorInteface {
  struct ONNXObject parent;
  void*(ONNXRUNTIME_API_STATUSCALL* Alloc)(void* this_, size_t size);
  void(ONNXRUNTIME_API_STATUSCALL* Free)(void* this_, void* p);
  const ONNXRuntimeAllocatorInfo*(ONNXRUNTIME_API_STATUSCALL* Info)(void* this_);
} ONNXRuntimeAllocatorInteface;

typedef ONNXRuntimeAllocatorInteface* ONNXRuntimeAllocator;


