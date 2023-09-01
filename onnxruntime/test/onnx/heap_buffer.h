// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>
#include <stdlib.h>
#include <stdint.h>
#include "callback.h"
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
    auto p = std::make_unique<uint8_t[]>(size);
    void* ret = p.get();
    buffers_.emplace_back(std::move(p));
    return ret;
  }
  void AddDeleter(const OrtCallback& d);

  HeapBuffer(const HeapBuffer&) = delete;
  HeapBuffer& operator=(const HeapBuffer&) = delete;

 private:
  std::vector<OrtCallback> deleters_;
  std::vector<std::unique_ptr<uint8_t[]> > buffers_;
};

}  // namespace test
}  // namespace onnxruntime
