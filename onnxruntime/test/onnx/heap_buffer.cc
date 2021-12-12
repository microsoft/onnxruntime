// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "heap_buffer.h"
#include "core/session/onnxruntime_c_api.h"
#include "callback.h"

namespace onnxruntime {
namespace test {
void HeapBuffer::AddDeleter(const OrtCallback& d) {
  deleters_.push_back(d);
}

HeapBuffer::~HeapBuffer() {
  for (auto d : deleters_) {
    d.Run();
  }
  for (void* p : buffers_) {
    delete[] reinterpret_cast<uint8_t*>(p);
  }
}
}  // namespace test
}  // namespace onnxruntime
