// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocatormgr.h"
#include "core/providers/nuphar/nuphar_allocator.h"
#include "test/framework/test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
TEST(AllocatorTest, NupharAllocatorTest) {
  int device_id = 0;
  TVMContext tvm_ctx;
  tvm_ctx.device_id = device_id;
  tvm_ctx.device_type = kDLCPU;
  DeviceAllocatorRegistrationInfo allocator_info(
      {OrtMemTypeDefault,
       [tvm_ctx](int /*id*/) { return std::make_unique<NupharAllocator>(tvm_ctx); },
       std::numeric_limits<size_t>::max()});

  auto nuphar_arena = CreateAllocator(allocator_info, device_id);

  EXPECT_STREQ(nuphar_arena->Info().name, TVM_STACKVM);
  EXPECT_EQ(nuphar_arena->Info().id, device_id);
  EXPECT_EQ(nuphar_arena->Info().mem_type, OrtMemTypeDefault);
  EXPECT_EQ(nuphar_arena->Info().type, OrtAllocatorType::OrtArenaAllocator);

  //test nuphar allocation
  size_t size = 1024;
  auto addr = nuphar_arena->Alloc(size);
  EXPECT_TRUE(addr);

  nuphar_arena->Free(addr);
}
}  // namespace test
}  // namespace onnxruntime
