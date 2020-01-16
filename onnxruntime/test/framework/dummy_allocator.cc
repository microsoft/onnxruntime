// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dummy_allocator.h"
#include "core/session/onnxruntime_cxx_api.h"
namespace onnxruntime {
namespace test {

DummyAllocator::DummyAllocator()
    : IAllocator(OrtMemoryInfo(kDummyAllocator, OrtAllocatorType::OrtDeviceAllocator)) {
}

void* DummyAllocator::Alloc(size_t size) {
  return malloc(size);
}

void DummyAllocator::Free(void* p) {
  free(p);
}

}  // namespace test
}  // namespace onnxruntime
