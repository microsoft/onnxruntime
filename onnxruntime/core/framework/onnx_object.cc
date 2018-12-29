// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include <atomic>

ORT_API(uint32_t, OrtAddRefToObject, void* ptr) {
  return (*static_cast<OrtObject**>(ptr))->AddRef(ptr);
}

ORT_API(uint32_t, OrtReleaseObject, void* ptr) {
  if (ptr == nullptr) return 0;
  return (*static_cast<OrtObject**>(ptr))->Release(ptr);
}

namespace {
struct ObjectImpl {
  const OrtObject* const cls;
  std::atomic_int ref_count;
};
}  // namespace
