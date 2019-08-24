// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <memory>
#if !(_MSC_VER)
#include <stdlib.h>
#endif

namespace onnxruntime {
namespace test {
struct OrtCallback;
/**
 * A holder for delay freed buffers
 */
class HeapBuffer {
 public:
  HeapBuffer() = default;
  /**
   * free all the buffers allocated from 'AllocMemory' function
   */
  ~HeapBuffer();
  void* AllocMemory(size_t size) {
    if (size == 0)
      return nullptr;

    void* p;
    size_t alignment = 64;
#if _MSC_VER
    p = _aligned_malloc(size, alignment);
    if (p == nullptr) throw std::bad_alloc();
#else
    int ret = posix_memalign(&p, alignment, size);
    if (ret != 0) throw std::bad_alloc();
#endif

    buffers_.push_back(p);
    return p;
  }
  void AddDeleter(OrtCallback* d);

 private:
  std::vector<OrtCallback*> deleters_;
  std::vector<void*> buffers_;
};
}  // namespace test
}  // namespace onnxruntime
