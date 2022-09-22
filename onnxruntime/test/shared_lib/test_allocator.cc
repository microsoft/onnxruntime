// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/cpu/cpu_provider_factory.h"
#include <gtest/gtest.h>


TEST(CApiTest, allocation_info) {
  auto cpu_mem_info_1 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto cpu_mem_info_2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  ASSERT_EQ(cpu_mem_info_1, cpu_mem_info_2);
  
  ASSERT_EQ(OrtMemoryInfoDeviceType::CPU, cpu_mem_info_1.GetDeviceType());
  ASSERT_EQ(OrtMemoryInfoDeviceType::CPU, cpu_mem_info_2.GetDeviceType());
  
  ASSERT_EQ("Cpu", cpu_mem_info_1.GetAllocatorName());
  ASSERT_EQ(OrtArenaAllocator, cpu_mem_info_1.GetAllocatorType());
  ASSERT_EQ(OrtMemTypeDefault, cpu_mem_info_1.GetMemoryType());
}

TEST(CApiTest, DefaultAllocator) {
  Ort::AllocatorWithDefaultOptions default_allocator;
  auto cpu_info = default_allocator.GetInfo();
  
  ASSERT_EQ("Cpu", cpu_info.GetAllocatorName());
  ASSERT_EQ(OrtMemoryInfoDeviceType::CPU, cpu_info.GetDeviceType());
  ASSERT_EQ(OrtDeviceAllocator, cpu_info.GetAllocatorType());
  ASSERT_EQ(OrtMemTypeDefault, cpu_info.GetMemoryType());

  Ort::MemoryAllocation allocation(default_allocator, default_allocator.Alloc(100), 100);
  ASSERT_EQ(allocation.size(), 100U);
  ASSERT_NE(allocation.get(), nullptr);
  memset(allocation.get(), 0, 100U);

  // Testing const wrapper around const pointer
  // one can call methods where C API allows const OrtAllocator
  // but not those where non-const is required
  const OrtAllocator* const_allocator_ptr = default_allocator;
  Ort::ConstAllocator const_allocator(const_allocator_ptr);
  auto const_mem_info = const_allocator.GetInfo();
  ASSERT_EQ(const_mem_info.GetAllocatorName(), "Cpu");
}

