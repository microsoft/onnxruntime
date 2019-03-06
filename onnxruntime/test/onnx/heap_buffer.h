// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <memory>
struct OrtCallback;

class HeapBuffer {
 public:
  HeapBuffer() = default;
  ~HeapBuffer();
  void* AllocMemory(size_t size) {
    void* p = malloc(size);
    buffers_.push_back(p);
    return p;
  }
  void AddDeleter(OrtCallback* d);

 private:
  std::vector<OrtCallback*> deleters_;
  std::vector<void*> buffers_;
};