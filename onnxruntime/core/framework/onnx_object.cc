// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include <atomic>

ONNXRUNTIME_API(uint32_t, ONNXRuntimeAddRefToObject, void* ptr) {
  return (*static_cast<ONNXObject**>(ptr))->AddRef(ptr);
}

ONNXRUNTIME_API(uint32_t, ONNXRuntimeReleaseObject, void* ptr) {
  if (ptr == nullptr) return 0;
  return (*static_cast<ONNXObject**>(ptr))->Release(ptr);
}

namespace {
struct ObjectImpl {
  const ONNXObject* const cls;
  std::atomic_int ref_count;
};
}  // namespace
