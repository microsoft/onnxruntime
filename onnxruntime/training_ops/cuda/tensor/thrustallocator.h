// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime {
namespace cuda {

class ThrustAllocator {
 public:
  typedef char value_type;

  ThrustAllocator(IAllocator* alloc) : alloc_(alloc) {}

  char* allocate(std::ptrdiff_t size) {
    return static_cast<char*>(alloc_->Alloc(size));
  }

  void deallocate(char* p, size_t /*size*/) {
    alloc_->Free(p);
  }

 private:
  IAllocator* alloc_;
};

}  // namespace cuda
}  // namespace onnxruntime
