// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif

namespace onnxruntime {

#ifdef USE_MIMALLOC
template <typename T>
using FastAllocVector = std::vector<T,mi_stl_allocator<T>>;
#else
template <typename T>
using FastAllocVector = std::vector<T>;
#endif 

}  // namespace onnxruntime
