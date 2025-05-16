// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/cpu/cpu_provider_factory.h"
#include <gtest/gtest.h>

#include "test/shared_lib/test_fixture.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace {
static constexpr PATH_TYPE MODEL_URI = TSTR("testdata/mul_1.onnx");
}

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

  // Default Allocator has no stats to report
  auto stats = default_allocator.GetStats();
  ASSERT_EQ(stats.size(), 0U);
}

TEST(CApiTest, SessionAllocator) {
  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, MODEL_URI, session_options);

  Ort::MemoryInfo infoCpu("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator allocator(session, infoCpu);

  std::vector<std::string> expected_keys = {
      "bytes_in_use",
      "bytes_limit",
      "num_allocs",
      "num_reserves",
      "num_arena_extensions",
      "num_arena_shrinkages",
      "total_allocated_bytes",
      "max_bytes_in_use",
      "max_alloc_size",
  };

  auto stats = allocator.GetStats();
  for (const auto& key : expected_keys) {
    ASSERT_TRUE(stats.find(key) != stats.end()) << "Missing key: " << key;
  }
}
