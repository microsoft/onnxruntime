// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_MIMALLOC_STL_ALLOCATOR)
#include <mimalloc.h>
#else
#include "core/framework/ort_stl_allocator.h"
#endif

namespace onnxruntime {

#if defined(USE_MIMALLOC_STL_ALLOCATOR)

template <typename T>
mi_stl_allocator<T> GetAllocator(const OpKernelContext& context) {
  ORT_UNUSED_PARAMETER(context);
  return mi_stl_allocator<T>();
}

template <typename T>
using FastAllocVector = std::vector<T,mi_stl_allocator<T>>;

#else

template <typename T>
OrtStlAllocator<T> GetAllocator(const OpKernelContext& context) {
  AllocatorPtr allocator;
  auto status = context.GetTempSpaceAllocator(&allocator);
  ORT_ENFORCE(status.IsOK());
  return OrtStlAllocator<T>(allocator);
}

template <typename T>
using FastAllocVector = std::vector<T,OrtStlAllocator<T>>;

#endif 

} // namespace onnxruntime
