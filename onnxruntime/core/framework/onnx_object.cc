// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/onnx_object.h"

ONNXRUNTIME_API(uint32_t, ONNXRuntimeAddRefToObject, void* ptr) {
  return (*static_cast<ONNXObject**>(ptr))->AddRef(ptr);
}
ONNXRUNTIME_API(uint32_t, ONNXRuntimeReleaseObject, void* ptr) {
  if (ptr == nullptr) return 0;
  return (*static_cast<ONNXObject**>(ptr))->Release(ptr);
}
