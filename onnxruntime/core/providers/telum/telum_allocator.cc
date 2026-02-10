// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telum_allocator.h"
#include <cstdlib>
#include <cstring>

namespace onnxruntime {
namespace telum {

void* TelumAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  // Allocate with 4K alignment as required by zDNN
  return AllocateAligned(size, ZDNN_ALIGNMENT);
}

void TelumAllocator::Free(void* p) {
  if (p != nullptr) {
    FreeAligned(p);
  }
}

void* TelumAllocator::AllocateAligned(size_t size, size_t alignment) {
  if (size == 0 || alignment == 0) {
    return nullptr;
  }

  // Ensure alignment is a power of 2
  if ((alignment & (alignment - 1)) != 0) {
    return nullptr;
  }

  void* ptr = nullptr;

#if defined(_WIN32)
  // Windows: use _aligned_malloc
  ptr = _aligned_malloc(size, alignment);
#else
  // POSIX: use posix_memalign
  if (posix_memalign(&ptr, alignment, size) != 0) {
    ptr = nullptr;
  }
#endif

  if (ptr != nullptr) {
    // Zero-initialize the memory for safety
    std::memset(ptr, 0, size);
  }

  return ptr;
}

void TelumAllocator::FreeAligned(void* p) {
  if (p == nullptr) {
    return;
  }

#if defined(_WIN32)
  _aligned_free(p);
#else
  free(p);
#endif
}

}  // namespace telum
}  // namespace onnxruntime

// Made with Bob
