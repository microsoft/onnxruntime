// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocatormgr.h"
#include "test/framework/test_utils.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace test {
TEST(AllocatorTest, CUDAAllocatorTest) {
  int cuda_device_id = 0;
  DeviceAllocatorRegistrationInfo default_allocator_info({OrtMemTypeDefault,
                                                          [](int id) { return std::make_unique<CUDAAllocator>(id, CUDA); }, std::numeric_limits<size_t>::max()});

  auto cuda_arena = CreateAllocator(default_allocator_info, cuda_device_id);

  size_t size = 1024;

  EXPECT_STREQ(cuda_arena->Info().name, CUDA);
  EXPECT_EQ(cuda_arena->Info().id, cuda_device_id);
  EXPECT_EQ(cuda_arena->Info().mem_type, OrtMemTypeDefault);
  EXPECT_EQ(cuda_arena->Info().type, OrtArenaAllocator);

  //test cuda allocation
  auto cuda_addr = cuda_arena->Alloc(size);
  EXPECT_TRUE(cuda_addr);

  DeviceAllocatorRegistrationInfo pinned_allocator_info({OrtMemTypeCPUOutput,
                                                         [](int) { return std::make_unique<CUDAPinnedAllocator>(0, CUDA_PINNED); }, std::numeric_limits<size_t>::max()});

  auto pinned_allocator = CreateAllocator(pinned_allocator_info);

  EXPECT_STREQ(pinned_allocator->Info().name, CUDA_PINNED);
  EXPECT_EQ(pinned_allocator->Info().id, 0);
  EXPECT_EQ(pinned_allocator->Info().mem_type, OrtMemTypeCPUOutput);
  EXPECT_EQ(pinned_allocator->Info().type, OrtArenaAllocator);

  //test pinned allocation
  auto pinned_addr = pinned_allocator->Alloc(size);
  EXPECT_TRUE(pinned_addr);

  const auto& cpu_arena = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
  EXPECT_STREQ(cpu_arena->Info().name, CPU);
  EXPECT_EQ(cpu_arena->Info().id, 0);
  EXPECT_EQ(cpu_arena->Info().mem_type, OrtMemTypeDefault);
  EXPECT_EQ(cpu_arena->Info().type, OrtArenaAllocator);

  auto cpu_addr_a = cpu_arena->Alloc(size);
  EXPECT_TRUE(cpu_addr_a);
  auto cpu_addr_b = cpu_arena->Alloc(size);
  EXPECT_TRUE(cpu_addr_b);
  memset(cpu_addr_a, -1, 1024);

  //test host-device memory copy
  cudaMemcpy(cuda_addr, cpu_addr_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(cpu_addr_b, cuda_addr, size, cudaMemcpyDeviceToHost);
  EXPECT_EQ(*((int*)cpu_addr_b), -1);

  cudaMemcpyAsync(pinned_addr, cuda_addr, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  EXPECT_EQ(*((int*)pinned_addr), -1);

  cpu_arena->Free(cpu_addr_a);
  cpu_arena->Free(cpu_addr_b);
  cuda_arena->Free(cuda_addr);
  pinned_allocator->Free(pinned_addr);
}
}  // namespace test
}  // namespace onnxruntime
