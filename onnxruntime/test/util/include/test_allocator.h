// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <atomic>
#include <stdexcept>
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include <assert.h>

struct MockedOrtAllocator : OrtAllocator {
  MockedOrtAllocator();
  ~MockedOrtAllocator();

  void* Alloc(size_t size);
  void Free(void* p);
  const OrtMemoryInfo* Info() const;
  void* Reserve(size_t size);
  size_t NumAllocations() const;
  size_t NumReserveAllocations() const;

  void LeakCheck();

 private:
  MockedOrtAllocator(const MockedOrtAllocator&) = delete;
  MockedOrtAllocator& operator=(const MockedOrtAllocator&) = delete;

  std::atomic<size_t> memory_inuse{0};
  std::atomic<size_t> num_allocations{0};
  std::atomic<size_t> num_reserve_allocations{0};
  OrtMemoryInfo* cpu_memory_info;
};
