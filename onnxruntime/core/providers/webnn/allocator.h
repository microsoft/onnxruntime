// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <emscripten.h>
#include <emscripten/val.h>

#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"

namespace onnxruntime {
namespace webnn {

class WebNNBufferAllocator : public IAllocator {
 public:
  WebNNBufferAllocator() : IAllocator(OrtMemoryInfo(WEBNN_BUFFER, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0), 0, OrtMemTypeDefault)) {}

  void* Alloc(size_t size) override;

  void Free(void* p) override;

  void GetStats(AllocatorStats* stats) override;

 private:
  AllocatorStats stats_;
  InlinedHashMap<void*, size_t> allocations_;
};

}  // namespace webnn
}  // namespace onnxruntime
