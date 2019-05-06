// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/cpu_provider_factory.h"
#include "test_fixture.h"

using namespace onnxruntime;

TEST_F(CApiTest, allocation_info) {
  OrtAllocatorInfo *info1, *info2;
  ORT_THROW_ON_ERROR(OrtCreateAllocatorInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault, &info1));
  ORT_THROW_ON_ERROR(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &info2));
  ASSERT_EQ(0, OrtCompareAllocatorInfo(info1, info2));
  OrtReleaseAllocatorInfo(info1);
  OrtReleaseAllocatorInfo(info2);
}

TEST_F(CApiTest, DefaultAllocator) {
  Ort::Allocator default_allocator = Ort::Allocator::Create_Default();
  char* p = (char*)default_allocator.Alloc(100);
  ASSERT_NE(p, nullptr);
  memset(p, 0, 100);
  default_allocator.Free(p);
  const OrtAllocatorInfo* info1 = default_allocator.GetInfo();
  const OrtAllocatorInfo* info2 = static_cast<OrtAllocator*>(default_allocator)->Info(default_allocator);
  ASSERT_EQ(0, OrtCompareAllocatorInfo(info1, info2));
}
