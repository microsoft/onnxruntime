// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_MIMALLOC_STL_ALLOCATOR)
#include <mimalloc.h>
#endif

namespace onnxruntime {

#if defined(USE_MIMALLOC_STL_ALLOCATOR)
template <typename T>
using FastAllocVector = std::vector<T,mi_stl_allocator<T>>;
#else
template <typename T>
using FastAllocVector = std::vector<T>;
#endif 

}  // namespace onnxruntime
