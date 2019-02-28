// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <memory>
struct OrtDeleter;

class HeapBuffer {
 public:
  HeapBuffer() = default;
  ~HeapBuffer();
  void* AllocMemory(size_t size) {
    void* p = malloc(size);
    buffers.push_back(p);
    return p;
  }
  void AddDeleter(OrtDeleter* d);

 private:
  std::vector<OrtDeleter*> deleters_;
  std::vector<void*> buffers;
};