// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/cpu/cpu_provider_factory.h"
#include "test/shared_lib/test_fixture.h"
#include "test/util/include/test_allocator.h"
#include <gtest/gtest.h>

extern std::unique_ptr<Ort::Env> ort_env;

TEST(CApiTest, allocation_info) {
  auto cpu_mem_info_1 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto cpu_mem_info_2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  ASSERT_EQ(cpu_mem_info_1, cpu_mem_info_2);

  ASSERT_EQ(OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU, cpu_mem_info_1.GetDeviceType());
  ASSERT_EQ(OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU, cpu_mem_info_2.GetDeviceType());

  ASSERT_EQ("Cpu", cpu_mem_info_1.GetAllocatorName());
  ASSERT_EQ(OrtArenaAllocator, cpu_mem_info_1.GetAllocatorType());
  ASSERT_EQ(OrtMemTypeDefault, cpu_mem_info_1.GetMemoryType());
}

TEST(CApiTest, DefaultAllocator) {
  Ort::AllocatorWithDefaultOptions default_allocator;
  auto cpu_info = default_allocator.GetInfo();

  ASSERT_EQ("Cpu", cpu_info.GetAllocatorName());
  ASSERT_EQ(OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU, cpu_info.GetDeviceType());
  ASSERT_EQ(OrtDeviceAllocator, cpu_info.GetAllocatorType());
  ASSERT_EQ(OrtMemTypeDefault, cpu_info.GetMemoryType());

  Ort::MemoryAllocation allocation(default_allocator, default_allocator.Alloc(100), 100);
  ASSERT_EQ(allocation.size(), 100U);
  ASSERT_NE(allocation.get(), nullptr);
  memset(allocation.get(), 0, 100U);

  // Default Allocator does not implement GetStats, we expect the stats to be empty.
  Ort::KeyValuePairs stats = default_allocator.GetStats();
  ASSERT_EQ(0, stats.GetKeyValuePairs().size());
}

#if !defined(ORT_MINIMAL_BUILD)
TEST(CApiTest, CustomAllocator) {
  constexpr PATH_TYPE model_path = TSTR("testdata/mul_1.onnx");

  const auto& api = Ort::GetApi();

  // Case 1: Register a custom allocator.
  {
    MockedOrtAllocator mocked_allocator;
    ASSERT_TRUE(api.RegisterAllocator(*ort_env, &mocked_allocator) == nullptr);

    Ort::SessionOptions session_options;
    session_options.AddConfigEntry("session.use_env_allocators", "1");
    Ort::Session session(*ort_env, model_path, session_options);

    Ort::Allocator allocator(session, mocked_allocator.Info());

    auto stats = allocator.GetStats();
    ASSERT_EQ(mocked_allocator.NumAllocations(), std::stoll(stats.GetValue("NumAllocs")));
    ASSERT_EQ(mocked_allocator.NumReserveAllocations(), std::stoll(stats.GetValue("NumReserves")));

    ASSERT_TRUE(api.UnregisterAllocator(*ort_env, mocked_allocator.Info()) == nullptr);
  }

  // Case 2: Register a custom allocator with an older API version which does not support GetStats.
  {
    MockedOrtAllocator mocked_allocator;
    mocked_allocator.version = 22;
    ASSERT_TRUE(api.RegisterAllocator(*ort_env, &mocked_allocator) == nullptr);

    Ort::SessionOptions session_options;
    session_options.AddConfigEntry("session.use_env_allocators", "1");
    Ort::Session session(*ort_env, model_path, session_options);

    Ort::Allocator allocator(session, mocked_allocator.Info());

    // Custom allocator does not implement GetStats, we expect the stats to be empty.
    auto stats = allocator.GetStats();
    ASSERT_EQ(0, stats.GetKeyValuePairs().size());

    ASSERT_TRUE(api.UnregisterAllocator(*ort_env, mocked_allocator.Info()) == nullptr);
  }
}
#endif
