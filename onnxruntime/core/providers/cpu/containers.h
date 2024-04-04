// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/ort_stl_allocator.h"

namespace onnxruntime {

template <typename T>
OrtStlAllocator<T> GetAllocator(const OpKernelContext& context) {
  AllocatorPtr allocator;
  auto status = context.GetTempSpaceAllocator(&allocator);
  ORT_ENFORCE(status.IsOK());
  return OrtStlAllocator<T>(allocator);
}

template <typename T>
using FastAllocVector = std::vector<T, OrtStlAllocator<T>>;

}  // namespace onnxruntime
