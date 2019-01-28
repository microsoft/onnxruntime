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
  std::unique_ptr<OrtAllocator> default_allocator;
  {
    OrtAllocator* ptr;
    ORT_THROW_ON_ERROR(OrtCreateDefaultAllocator(&ptr));
    default_allocator.reset(ptr);
  }
  char* p = (char*)OrtAllocatorAlloc(default_allocator.get(), 100);
  ASSERT_NE(p, nullptr);
  memset(p, 0, 100);
  OrtAllocatorFree(default_allocator.get(), p);
  const OrtAllocatorInfo* info1 = OrtAllocatorGetInfo(default_allocator.get());
  const OrtAllocatorInfo* info2 = default_allocator->Info(default_allocator.get());
  ASSERT_EQ(0, OrtCompareAllocatorInfo(info1, info2));
}
