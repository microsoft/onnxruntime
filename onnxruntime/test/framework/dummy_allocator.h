// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace test {
struct DummyAllocator : IAllocator {
  static constexpr const char* kDummyAllocator = "DummyAllocator";

  DummyAllocator();
  ~DummyAllocator() = default;

  void* Alloc(size_t size) override;
  void Free(void* p) override;
  const OrtMemoryInfo& Info() const override { return allocator_info_; }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DummyAllocator);

  OrtMemoryInfo allocator_info_;
};
}  // namespace test
}  // namespace onnxruntime
