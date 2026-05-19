// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator_utils.h"
#include "gtest/gtest.h"
#include "cuda_runtime.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/shared_inc/gpu_external_memory_allocator.h"

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace onnxruntime {
namespace test {
namespace {
std::atomic<size_t> test_external_alloc_count{0};
std::atomic<size_t> test_external_free_count{0};
std::atomic<size_t> test_external_empty_cache_count{0};

void* TestExternalAlloc(size_t size) {
  test_external_alloc_count.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(size);
}

void TestExternalFree(void* p) {
  test_external_free_count.fetch_add(1, std::memory_order_relaxed);
  std::free(p);
}

void TestExternalEmptyCache() {
  test_external_empty_cache_count.fetch_add(1, std::memory_order_relaxed);
}

void ResetTestExternalAllocatorCallbackCounts() {
  test_external_alloc_count.store(0, std::memory_order_relaxed);
  test_external_free_count.store(0, std::memory_order_relaxed);
  test_external_empty_cache_count.store(0, std::memory_order_relaxed);
}

struct CudaFreeDeleter {
  void operator()(void* p) const {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
};

bool IsCudaDeviceUnavailable() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    return true;
  }

  return false;
}
}  // namespace

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

TEST(AllocatorTest, CUDAExternalMemoryAllocatorTest) {
  std::vector<char> external_memory(1024);
  GpuExternalMemoryAllocator allocator(0, CUDA, external_memory.data(), external_memory.size());

  void* first = allocator.Alloc(128);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(first) % 256, static_cast<uintptr_t>(0));

  void* second = allocator.Alloc(128);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first, second);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(second) % 256, static_cast<uintptr_t>(0));

  allocator.Free(first);
  void* reused = allocator.Alloc(64);
  EXPECT_EQ(first, reused);

  void* reserved = allocator.Reserve(32);
  ASSERT_NE(reserved, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(reserved) % 256, static_cast<uintptr_t>(0));

  allocator.Free(second);
  allocator.Free(reused);
  allocator.Free(reserved);
}

TEST(AllocatorTest, CUDAExternalMemoryAllocatorExhaustionTest) {
  std::vector<char> external_memory(512 + 255);
  const auto aligned_address = (reinterpret_cast<uintptr_t>(external_memory.data()) + 255) & ~static_cast<uintptr_t>(255);
  auto* external_mem_ptr = reinterpret_cast<void*>(aligned_address);
  CUDAExternalMemoryAllocator allocator(0, CUDA, external_mem_ptr, 512);

  void* allocation = allocator.Alloc(512);
  ASSERT_NE(allocation, nullptr);
  EXPECT_THROW(allocator.Alloc(1), std::exception);

  allocator.Free(allocation);
}

TEST(AllocatorTest, CUDAExternalMemoryAllocatorCoalescesFreedBlocks) {
  std::vector<char> external_memory(1024 + 255);
  const auto aligned_address = (reinterpret_cast<uintptr_t>(external_memory.data()) + 255) & ~static_cast<uintptr_t>(255);
  auto* external_mem_ptr = reinterpret_cast<void*>(aligned_address);
  CUDAExternalMemoryAllocator allocator(0, CUDA, external_mem_ptr, 1024);

  void* first = allocator.Alloc(128);
  ASSERT_NE(first, nullptr);
  void* second = allocator.Alloc(128);
  ASSERT_NE(second, nullptr);
  void* third = allocator.Alloc(128);
  ASSERT_NE(third, nullptr);

  allocator.Free(second);
  allocator.Free(first);
  allocator.Free(third);

  void* whole_buffer = allocator.Alloc(1024);
  ASSERT_NE(whole_buffer, nullptr);
  EXPECT_EQ(whole_buffer, external_mem_ptr);

  allocator.Free(whole_buffer);
}

TEST(AllocatorTest, CUDAExternalMemoryProviderOptionsRoundTrip) {
  std::vector<char> external_memory(1024);
  void* external_mem_ptr = external_memory.data();
  const size_t external_mem_size = external_memory.size();
  void* external_alloc = reinterpret_cast<void*>(&TestExternalAlloc);
  void* external_free = reinterpret_cast<void*>(&TestExternalFree);
  void* external_empty_cache = reinterpret_cast<void*>(&TestExternalEmptyCache);

  const ProviderOptions input_options{
      {"gpu_external_alloc", std::to_string(reinterpret_cast<size_t>(external_alloc))},
      {"gpu_external_free", std::to_string(reinterpret_cast<size_t>(external_free))},
      {"gpu_external_empty_cache", std::to_string(reinterpret_cast<size_t>(external_empty_cache))},
      {"gpu_external_mem_ptr", std::to_string(reinterpret_cast<size_t>(external_mem_ptr))},
      {"gpu_external_mem_size", std::to_string(external_mem_size)},
  };

  const auto output_options =
      CUDAExecutionProviderInfo::ToProviderOptions(CUDAExecutionProviderInfo::FromProviderOptions(input_options));

  for (const auto& [key, value] : input_options) {
    const auto output_iter = output_options.find(key);
    ASSERT_NE(output_iter, output_options.end()) << key;
    EXPECT_EQ(value, output_iter->second) << key;
  }
}

TEST(AllocatorTest, CUDAExternalAllocatorInfoRejectsInvalidConfig) {
  std::vector<char> external_memory(4096);

  CUDAExecutionProviderExternalAllocatorInfo no_config{};
  EXPECT_TRUE(no_config.ValidateExternalAllocatorConfig(false).IsOK());
  EXPECT_FALSE(no_config.UsesExternalDeviceAllocator());

  CUDAExecutionProviderExternalAllocatorInfo memory_config{};
  memory_config.mem_ptr = external_memory.data();
  memory_config.mem_size = external_memory.size();
  EXPECT_TRUE(memory_config.ValidateExternalAllocatorConfig(false).IsOK());
  EXPECT_TRUE(memory_config.UsesExternalDeviceAllocator());
  EXPECT_FALSE(memory_config.ValidateExternalAllocatorConfig(true).IsOK());

  CUDAExecutionProviderExternalAllocatorInfo pointer_only{};
  pointer_only.mem_ptr = external_memory.data();
  EXPECT_FALSE(pointer_only.ValidateExternalAllocatorConfig(false).IsOK());

  CUDAExecutionProviderExternalAllocatorInfo size_only{};
  size_only.mem_size = external_memory.size();
  EXPECT_FALSE(size_only.ValidateExternalAllocatorConfig(false).IsOK());

  CUDAExecutionProviderExternalAllocatorInfo callback_only{};
  callback_only.alloc = reinterpret_cast<void*>(TestExternalAlloc);
  EXPECT_FALSE(callback_only.ValidateExternalAllocatorConfig(false).IsOK());

  CUDAExecutionProviderExternalAllocatorInfo mixed_config{};
  mixed_config.alloc = reinterpret_cast<void*>(TestExternalAlloc);
  mixed_config.free = reinterpret_cast<void*>(TestExternalFree);
  mixed_config.mem_ptr = external_memory.data();
  mixed_config.mem_size = external_memory.size();
  EXPECT_FALSE(mixed_config.ValidateExternalAllocatorConfig(false).IsOK());
}

TEST(AllocatorTest, CreateCudaAllocatorRejectsInvalidExternalMemoryConfig) {
  std::vector<char> external_memory(4096);
  CUDAExecutionProvider::CUDAAllocatorParams params{};
  CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};
  params.external_alloc_info = &external_allocator_info;

  external_allocator_info.mem_ptr = external_memory.data();
  EXPECT_THROW(CUDAExecutionProvider::CreateCudaAllocator(params), std::exception);

  external_allocator_info.mem_ptr = nullptr;
  external_allocator_info.mem_size = external_memory.size();
  EXPECT_THROW(CUDAExecutionProvider::CreateCudaAllocator(params), std::exception);

  external_allocator_info.alloc = reinterpret_cast<void*>(TestExternalAlloc);
  external_allocator_info.free = reinterpret_cast<void*>(TestExternalFree);
  external_allocator_info.mem_ptr = external_memory.data();
  external_allocator_info.mem_size = external_memory.size();
  EXPECT_THROW(CUDAExecutionProvider::CreateCudaAllocator(params), std::exception);

  external_allocator_info = {};
  external_allocator_info.alloc = reinterpret_cast<void*>(TestExternalAlloc);
  EXPECT_THROW(CUDAExecutionProvider::CreateCudaAllocator(params), std::exception);

  external_allocator_info = {};
  external_allocator_info.free = reinterpret_cast<void*>(TestExternalFree);
  EXPECT_THROW(CUDAExecutionProvider::CreateCudaAllocator(params), std::exception);

  external_allocator_info = {};
  external_allocator_info.empty_cache = reinterpret_cast<void*>(TestExternalEmptyCache);
  EXPECT_THROW(CUDAExecutionProvider::CreateCudaAllocator(params), std::exception);

  external_allocator_info = {};
  external_allocator_info.alloc = reinterpret_cast<void*>(TestExternalAlloc);
  external_allocator_info.mem_ptr = external_memory.data();
  external_allocator_info.mem_size = external_memory.size();
  EXPECT_THROW(CUDAExecutionProvider::CreateCudaAllocator(params), std::exception);
}

TEST(AllocatorTest, CreateCudaAllocatorUsesExternalMemory) {
  std::vector<char> external_memory(4096);
  CUDAExecutionProvider::CUDAAllocatorParams params{};
  CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};
  external_allocator_info.mem_ptr = external_memory.data();
  external_allocator_info.mem_size = external_memory.size();
  params.external_alloc_info = &external_allocator_info;

  const auto allocator = CUDAExecutionProvider::CreateCudaAllocator(params);
  ASSERT_NE(allocator, nullptr);
  EXPECT_EQ(allocator->Info().alloc_type, OrtDeviceAllocator);

  void* allocation = allocator->Alloc(1024);
  ASSERT_NE(allocation, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(allocation) % 256, static_cast<uintptr_t>(0));

  allocator->Free(allocation);
}

TEST(AllocatorTest, CreateCudaAllocatorReserveUsesExternalMemory) {
  std::vector<char> external_memory(2048 + 255);
  const auto aligned_address = (reinterpret_cast<uintptr_t>(external_memory.data()) + 255) & ~static_cast<uintptr_t>(255);
  auto* external_mem_ptr = reinterpret_cast<void*>(aligned_address);

  CUDAExecutionProvider::CUDAAllocatorParams params{};
  CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};
  external_allocator_info.mem_ptr = external_mem_ptr;
  external_allocator_info.mem_size = 2048;
  params.external_alloc_info = &external_allocator_info;

  const auto allocator = CUDAExecutionProvider::CreateCudaAllocator(params);
  ASSERT_NE(allocator, nullptr);
  EXPECT_EQ(allocator->Info().alloc_type, OrtDeviceAllocator);

  void* reserved = allocator->Reserve(512);
  ASSERT_NE(reserved, nullptr);
  const auto reserved_address = reinterpret_cast<uintptr_t>(reserved);
  const auto base = reinterpret_cast<uintptr_t>(external_mem_ptr);
  EXPECT_GE(reserved_address, base);
  EXPECT_LT(reserved_address, base + external_allocator_info.mem_size);
  EXPECT_EQ(reserved_address % 256, static_cast<uintptr_t>(0));

  allocator->Free(reserved);
}

TEST(AllocatorTest, CUDAExternalAllocatorReserveFreeEmptiesCache) {
  ResetTestExternalAllocatorCallbackCounts();
  CUDAExternalAllocator allocator(0, CUDA, reinterpret_cast<void*>(TestExternalAlloc),
                                  reinterpret_cast<void*>(TestExternalFree),
                                  reinterpret_cast<void*>(TestExternalEmptyCache));

  void* allocation = allocator.Alloc(64);
  ASSERT_NE(allocation, nullptr);
  allocator.Free(allocation);
  EXPECT_EQ(test_external_alloc_count.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(test_external_free_count.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(test_external_empty_cache_count.load(std::memory_order_relaxed), 0U);

  void* reserved = allocator.Reserve(64);
  ASSERT_NE(reserved, nullptr);
  allocator.Free(reserved);
  EXPECT_EQ(test_external_alloc_count.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(test_external_free_count.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(test_external_empty_cache_count.load(std::memory_order_relaxed), 1U);
}

TEST(AllocatorTest, CreateCudaAllocatorUsesCudaExternalMemory) {
  if (IsCudaDeviceUnavailable()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  OrtDevice::DeviceId cuda_device_id = 0;
  CUDA_CALL_THROW(cudaSetDevice(cuda_device_id));

  const size_t external_mem_size = 4096;
  void* external_mem_ptr = nullptr;
  CUDA_CALL_THROW(cudaMalloc(&external_mem_ptr, external_mem_size));
  std::unique_ptr<void, CudaFreeDeleter> external_memory{external_mem_ptr};

  CUDAExecutionProvider::CUDAAllocatorParams params{};
  CUDAExecutionProviderExternalAllocatorInfo external_allocator_info{};
  external_allocator_info.mem_ptr = external_mem_ptr;
  external_allocator_info.mem_size = external_mem_size;
  params.external_alloc_info = &external_allocator_info;

  const auto allocator = CUDAExecutionProvider::CreateCudaAllocator(params);
  ASSERT_NE(allocator, nullptr);

  void* first_allocation = allocator->Alloc(1024);
  ASSERT_NE(first_allocation, nullptr);
  const auto base = reinterpret_cast<uintptr_t>(external_mem_ptr);
  const auto first_address = reinterpret_cast<uintptr_t>(first_allocation);
  EXPECT_GE(first_address, base);
  EXPECT_LT(first_address, base + external_mem_size);
  EXPECT_EQ(first_address % 256, static_cast<uintptr_t>(0));
  CUDA_CALL_THROW(cudaMemset(first_allocation, 0, 1024));

  void* reserved_allocation = allocator->Reserve(512);
  ASSERT_NE(reserved_allocation, nullptr);
  const auto reserved_address = reinterpret_cast<uintptr_t>(reserved_allocation);
  EXPECT_GE(reserved_address, base);
  EXPECT_LT(reserved_address, base + external_mem_size);
  EXPECT_EQ(reserved_address % 256, static_cast<uintptr_t>(0));
  CUDA_CALL_THROW(cudaMemset(reserved_allocation, 0, 512));

  allocator->Free(reserved_allocation);
  allocator->Free(first_allocation);
}

TEST(AllocatorTest, CUDAExecutionProviderRejectsInvalidExternalAllocatorConfigBeforeCudaInit) {
  std::vector<char> external_memory(4096);

  CUDAExecutionProviderInfo pointer_only{};
  pointer_only.external_allocator_info.mem_ptr = external_memory.data();
  EXPECT_THROW(CUDAExecutionProvider execution_provider(pointer_only), std::exception);

  CUDAExecutionProviderInfo memory_with_user_stream{};
  memory_with_user_stream.has_user_compute_stream = true;
  memory_with_user_stream.user_compute_stream = reinterpret_cast<void*>(0x1234);
  memory_with_user_stream.external_allocator_info.mem_ptr = external_memory.data();
  memory_with_user_stream.external_allocator_info.mem_size = external_memory.size();
  EXPECT_THROW(CUDAExecutionProvider execution_provider(memory_with_user_stream), std::exception);

  CUDAExecutionProviderInfo mixed_config{};
  mixed_config.external_allocator_info.alloc = reinterpret_cast<void*>(TestExternalAlloc);
  mixed_config.external_allocator_info.free = reinterpret_cast<void*>(TestExternalFree);
  mixed_config.external_allocator_info.mem_ptr = external_memory.data();
  mixed_config.external_allocator_info.mem_size = external_memory.size();
  EXPECT_THROW(CUDAExecutionProvider execution_provider(mixed_config), std::exception);
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
}  // namespace test
}  // namespace onnxruntime
