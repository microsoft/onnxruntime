// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator_utils.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/plugin/cuda_plugin_temp_space_allocator.h"

#include <stdexcept>

namespace onnxruntime {
namespace test {

TEST(AllocatorTest, CUDAAllocatorTest) {
  OrtDevice::DeviceId cuda_device_id = 0;

  // ensure CUDA device is available.
  CUDA_CALL_THROW(cudaSetDevice(cuda_device_id));

  AllocatorCreationInfo default_memory_info(
      {[](OrtDevice::DeviceId id) { return std::make_unique<CUDAAllocator>(id, CUDA); }, cuda_device_id});

  auto cuda_arena = CreateAllocator(default_memory_info);

  size_t size = 1024;

  EXPECT_STREQ(cuda_arena->Info().name.c_str(), CUDA);
  EXPECT_EQ(cuda_arena->Info().device.Id(), cuda_device_id);
  EXPECT_EQ(cuda_arena->Info().mem_type, OrtMemTypeDefault);
  EXPECT_EQ(cuda_arena->Info().alloc_type, OrtArenaAllocator);

  // test cuda allocation
  auto cuda_addr = cuda_arena->Alloc(size);
  EXPECT_TRUE(cuda_addr);

  AllocatorCreationInfo pinned_memory_info(
      [](int device_id) { return std::make_unique<CUDAPinnedAllocator>(device_id, CUDA_PINNED); });

  auto pinned_allocator = CreateAllocator(pinned_memory_info);

  EXPECT_STREQ(pinned_allocator->Info().name.c_str(), CUDA_PINNED);
  EXPECT_EQ(pinned_allocator->Info().device.Id(), 0);
  EXPECT_EQ(pinned_allocator->Info().mem_type, OrtMemTypeCPUOutput);
  EXPECT_EQ(pinned_allocator->Info().alloc_type, OrtArenaAllocator);

  // test pinned allocation
  auto pinned_addr = pinned_allocator->Alloc(size);
  EXPECT_TRUE(pinned_addr);

  AllocatorCreationInfo cpu_memory_info(
      [](int) { return std::make_unique<CPUAllocator>(); }, true);
  const auto& cpu_arena = CreateAllocator(cpu_memory_info);
  EXPECT_STREQ(cpu_arena->Info().name.c_str(), CPU);
  EXPECT_EQ(cpu_arena->Info().device.Id(), 0);
  EXPECT_EQ(cpu_arena->Info().mem_type, OrtMemTypeDefault);
  EXPECT_EQ(cpu_arena->Info().alloc_type, OrtArenaAllocator);

  auto cpu_addr_a = cpu_arena->Alloc(size);
  EXPECT_TRUE(cpu_addr_a);
  auto cpu_addr_b = cpu_arena->Alloc(size);
  EXPECT_TRUE(cpu_addr_b);
  memset(cpu_addr_a, -1, 1024);

  // test host-device memory copy
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

// test that we fallback to smaller allocations if the growth of the arena exceeds the available memory
TEST(AllocatorTest, CUDAAllocatorFallbackTest) {
  OrtDevice::DeviceId cuda_device_id = 0;

  size_t free = 0;
  size_t total = 0;

  CUDA_CALL_THROW(cudaSetDevice(cuda_device_id));
  CUDA_CALL_THROW(cudaMemGetInfo(&free, &total));

  // need extra test logic if this ever happens.
  EXPECT_NE(free, total) << "All memory is free. Test logic does not handle this.";

  AllocatorCreationInfo default_memory_info(
      {[](OrtDevice::DeviceId id) { return std::make_unique<CUDAAllocator>(id, CUDA); },
       cuda_device_id});

  auto cuda_arena = CreateAllocator(default_memory_info);

  // initial allocation that sets the growth size for the next allocation
  size_t size = total / 2;
  void* cuda_addr_0 = cuda_arena->Alloc(size);
  EXPECT_TRUE(cuda_addr_0);

  // this should trigger an allocation equal to the current total, which should fail initially and gradually fall back
  // to a smaller block.
  size_t next_size = 1024;

  void* cuda_addr_1 = cuda_arena->Alloc(next_size);
  EXPECT_TRUE(cuda_addr_1);
  cuda_arena->Free(cuda_addr_0);
  cuda_arena->Free(cuda_addr_1);
  cuda_arena = nullptr;

  auto last_error = cudaGetLastError();
  EXPECT_EQ(last_error, cudaSuccess) << "Last error should be cleared if handled gracefully";
}

// Kernels request temporary buffers from the plugin's temp-space allocator. It has to charge the session's own budget
// account, otherwise those buffers escape the session limit.
TEST(AllocatorTest, CudaPluginTempSpaceAllocatorChargesSessionBudgetAccount) {
  constexpr OrtDevice::DeviceId cuda_device_id = 0;
  constexpr size_t kSessionLimit = 4096;

  CUDA_CALL_THROW(cudaSetDevice(cuda_device_id));

  auto account = std::make_shared<SessionBudgetAccount>(GetCudaDeviceMemoryBudget(cuda_device_id), kSessionLimit);
  CUDAAllocator session_allocator(cuda_device_id, CUDA, account);
  AllocatorPtr temp_allocator = cuda_plugin::CreateCudaPluginTempSpaceAllocator(cuda_device_id, account);

  ASSERT_EQ(static_cast<CUDAAllocator*>(temp_allocator.get())->GetBudgetAccount(), account);

  // Fill the session account through the session's own allocator.
  void* session_buffer = session_allocator.Alloc(kSessionLimit);
  ASSERT_NE(session_buffer, nullptr);
  ASSERT_EQ(account->GetAccountStats().committed, kSessionLimit);

  // The temp-space allocator now has to be rejected. With its own unlimited account it allocated 8192 bytes here.
  EXPECT_THROW(static_cast<void>(temp_allocator->Alloc(2 * kSessionLimit)), std::runtime_error);
  EXPECT_THROW(static_cast<void>(temp_allocator->Alloc(1)), std::runtime_error);
  EXPECT_EQ(account->GetAccountStats().committed, kSessionLimit);

  session_allocator.Free(session_buffer);
  ASSERT_EQ(account->GetAccountStats().committed, 0u);

  // Charges also flow the other way: temp-space allocations are visible to the session's allocator.
  void* temp_buffer = temp_allocator->Alloc(kSessionLimit);
  ASSERT_NE(temp_buffer, nullptr);
  EXPECT_EQ(account->GetAccountStats().committed, kSessionLimit);
  EXPECT_THROW(static_cast<void>(session_allocator.Alloc(1)), std::runtime_error);

  temp_allocator->Free(temp_buffer);
  EXPECT_EQ(account->GetAccountStats().committed, 0u);
}
}  // namespace test
}  // namespace onnxruntime
