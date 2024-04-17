// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "test_allocator.h"

MockedOrtAllocator::MockedOrtAllocator() {
  OrtAllocator::version = ORT_API_VERSION;
  OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<MockedOrtAllocator*>(this_)->Alloc(size); };
  OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<MockedOrtAllocator*>(this_)->Free(p); };
  OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const MockedOrtAllocator*>(this_)->Info(); };
  OrtAllocator::Reserve = [](OrtAllocator* this_, size_t size) { return static_cast<MockedOrtAllocator*>(this_)->Reserve(size); };
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_memory_info));
}

MockedOrtAllocator::~MockedOrtAllocator() {
  Ort::GetApi().ReleaseMemoryInfo(cpu_memory_info);
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 26400)
#pragma warning(disable : 26409)
#endif
void* MockedOrtAllocator::Alloc(size_t size) {
  constexpr size_t extra_len = sizeof(size_t);
  memory_inuse.fetch_add(size += extra_len);
  void* p = new (std::nothrow) uint8_t[size];
  if (p == nullptr)
    return p;
  num_allocations.fetch_add(1);
  *(size_t*)p = size;
  return (char*)p + extra_len;
}

void* MockedOrtAllocator::Reserve(size_t size) {
  constexpr size_t extra_len = sizeof(size_t);
  memory_inuse.fetch_add(size += extra_len);
  void* p = new (std::nothrow) uint8_t[size];
  if (p == nullptr)
    return p;
  num_allocations.fetch_add(1);
  num_reserve_allocations.fetch_add(1);
  *(size_t*)p = size;
  return (char*)p + extra_len;
}

void MockedOrtAllocator::Free(void* p) {
  constexpr size_t extra_len = sizeof(size_t);
  if (!p) return;
  p = (char*)p - extra_len;
  size_t len = *(size_t*)p;
  memory_inuse.fetch_sub(len);
  delete[] reinterpret_cast<uint8_t*>(p);
}

const OrtMemoryInfo* MockedOrtAllocator::Info() const {
  return cpu_memory_info;
}

size_t MockedOrtAllocator::NumAllocations() const {
  return num_allocations.load();
}

size_t MockedOrtAllocator::NumReserveAllocations() const {
  return num_reserve_allocations.load();
}

void MockedOrtAllocator::LeakCheck() {
  if (memory_inuse.load())
    ORT_THROW("memory leak!!!");
}
