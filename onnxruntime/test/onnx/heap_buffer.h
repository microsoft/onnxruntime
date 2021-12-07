// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "callback.h"

#include <vector>
#include <stdlib.h>
#include <stdint.h>

namespace onnxruntime {
namespace test {
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
    void* p = new (std::nothrow) uint8_t[size];
    buffers_.push_back(p);
    return p;
  }
  void AddDeleter(const OrtCallback& d);

  HeapBuffer(const HeapBuffer&) = delete;
  HeapBuffer& operator=(const HeapBuffer&) = delete;

 private:
  std::vector<OrtCallback> deleters_;
  std::vector<void*> buffers_;
};

}  // namespace test
}  // namespace onnxruntime
