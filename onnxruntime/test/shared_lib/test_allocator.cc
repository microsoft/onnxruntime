// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/providers/cpu/cpu_provider_factory.h"
#include <gtest/gtest.h>


TEST(CApiTest, allocation_info) {
  OrtMemoryInfo *info1, *info2;
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &info1));
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &info2));
  int result;
  Ort::ThrowOnError(Ort::GetApi().CompareMemoryInfo(info1, info2, &result));
  ASSERT_EQ(0, result);
  Ort::GetApi().ReleaseMemoryInfo(info1);
  Ort::GetApi().ReleaseMemoryInfo(info2);
}

TEST(CApiTest, DefaultAllocator) {
  Ort::AllocatorWithDefaultOptions default_allocator;
  char* p = (char*)default_allocator.Alloc(100);
  ASSERT_NE(p, nullptr);
  memset(p, 0, 100);
  default_allocator.Free(p);
  const OrtMemoryInfo* info1 = default_allocator.GetInfo();
  const OrtMemoryInfo* info2 = static_cast<OrtAllocator*>(default_allocator)->Info(default_allocator);
  int result;
  Ort::ThrowOnError(Ort::GetApi().CompareMemoryInfo(info1, info2, &result));
  ASSERT_EQ(0, result);
}
