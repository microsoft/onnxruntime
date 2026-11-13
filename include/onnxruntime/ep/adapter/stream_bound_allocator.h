// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <utility>

#include "core/framework/allocator.h"

namespace onnxruntime::ep::adapter {

// Kernel-local stream shims are not host OrtSyncStreams. Only the context's framework stream
// may be forwarded to the session allocator, including for callers that use plain Alloc().
class StreamBoundAllocator final : public IAllocator {
 public:
  StreamBoundAllocator(AllocatorPtr allocator, Stream* framework_stream)
      : IAllocator(allocator->Info()), allocator_(std::move(allocator)), stream_(framework_stream) {}

  void* Alloc(size_t size) override {
    return IsStreamAware() ? allocator_->AllocOnStream(size, stream_) : allocator_->Alloc(size);
  }

  bool IsStreamAware() const override {
    return stream_ != nullptr && allocator_->IsStreamAware();
  }

  void* AllocOnStream(size_t size, Stream* /*stream*/) override {
    return Alloc(size);
  }

  void Free(void* p) override {
    allocator_->Free(p);
  }

  void* Reserve(size_t size) override {
    return allocator_->Reserve(size);
  }

  void GetStats(AllocatorStats* stats) override {
    allocator_->GetStats(stats);
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(StreamBoundAllocator);
  AllocatorPtr allocator_;
  Stream* stream_;
};

}  // namespace onnxruntime::ep::adapter
