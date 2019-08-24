// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "heap_buffer.h"
#include "core/framework/utils.h"
#include "core/session/onnxruntime_c_api.h"
#include "callback.h"

namespace onnxruntime {
namespace test {
void HeapBuffer::AddDeleter(OrtCallback* d) {
  if (d != nullptr) deleters_.push_back(d);
}

void* HeapBuffer::AllocMemory(size_t size) {
  return utils::DefaultAlloc(size);
}

HeapBuffer::~HeapBuffer() {
  for (auto d : deleters_) {
    OrtRunCallback(d);
  }
  for (void* p : buffers_) {
    utils::DefaultFree(p);
  }
}
}  // namespace test
}  // namespace onnxruntime
