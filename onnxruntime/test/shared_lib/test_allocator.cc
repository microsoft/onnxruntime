// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/cpu_provider_factory.h"
#include "test_fixture.h"

using namespace onnxruntime;

TEST_F(CApiTest, allocation_info) {
  ONNXRuntimeAllocatorInfo *info1, *info2;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateAllocatorInfo("Cpu", ONNXRuntimeArenaAllocator, 0, ONNXRuntimeMemTypeDefault, &info1));
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateCpuAllocatorInfo(ONNXRuntimeArenaAllocator, ONNXRuntimeMemTypeDefault, &info2));
  ASSERT_EQ(0, ONNXRuntimeCompareAllocatorInfo(info1, info2));
  ReleaseONNXRuntimeAllocatorInfo(info1);
  ReleaseONNXRuntimeAllocatorInfo(info2);
}

TEST_F(CApiTest, DefaultAllocator) {
  std::unique_ptr<ONNXRuntimeAllocator> default_allocator;
  {
    ONNXRuntimeAllocator* ptr;
    ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateDefaultAllocator(&ptr));
    default_allocator.reset(ptr);
  }
  char* p = (char*)ONNXRuntimeAllocatorAlloc(default_allocator.get(), 100);
  ASSERT_NE(p, nullptr);
  memset(p, 0, 100);
  ONNXRuntimeAllocatorFree(default_allocator.get(), p);
  const ONNXRuntimeAllocatorInfo* info1 = ONNXRuntimeAllocatorGetInfo(default_allocator.get());
  const ONNXRuntimeAllocatorInfo* info2 = (*default_allocator)->Info(default_allocator.get());
  ASSERT_EQ(0, ONNXRuntimeCompareAllocatorInfo(info1, info2));
}
