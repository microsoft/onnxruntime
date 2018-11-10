// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/cpu_provider_factory.h"
#include "test_fixture.h"

using namespace onnxruntime;

TEST_F(CApiTest, allocation_info) {
  ONNXRuntimeAllocatorInfoPtr info1, info2;
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateAllocatorInfo("Cpu", ONNXRuntimeArenaAllocator, 0, ONNXRuntimeMemTypeDefault, &info1));
  ONNXRUNTIME_THROW_ON_ERROR(ONNXRuntimeCreateCpuAllocatorInfo(ONNXRuntimeArenaAllocator, ONNXRuntimeMemTypeDefault, &info2));
  ASSERT_EQ(0, ONNXRuntimeCompareAllocatorInfo(info1, info2));
  ReleaseONNXRuntimeAllocatorInfo(info1);
  ReleaseONNXRuntimeAllocatorInfo(info2);
}