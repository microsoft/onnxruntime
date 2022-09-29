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
  return new (std::nothrow) uint8_t[size];
}

void DummyAllocator::Free(void* p) {
  delete[] reinterpret_cast<uint8_t*>(p);
}

}  // namespace test
}  // namespace onnxruntime
