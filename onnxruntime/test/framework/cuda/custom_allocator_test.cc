//// Copyright (c) Microsoft Corporation. All rights reserved.
//// Licensed under the MIT License.
#include <memory>
#include "cuda_runtime.h"
#include "core/framework/execution_provider.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/provider_factory_creators.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
class CustomAllocator {
public:
  static CustomAllocator& GetInstance() {
    static CustomAllocator instance_;
    return instance_;
  };
  void* Alloc(size_t nbytes) {
    alloc_count_ += 1;
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, nbytes);
    if (err != cudaSuccess) {
      throw std::runtime_error("CustomAllocator::alloc failed: " + std::string(cudaGetErrorString(err)));
    }
    return ptr;
  }
  void Free(void* ptr) {
    free_count_ += 1;
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
      throw std::runtime_error("CustomAllocator::free failed: " + std::string(cudaGetErrorString(err)));
    }
  }
  size_t GetAllocCount() const {
    return alloc_count_;
  }
  size_t GetFreeCount() const {
    return free_count_;
  }
private:
  CustomAllocator() = default;
  ~CustomAllocator() = default;
  CustomAllocator(const CustomAllocator&) = delete;
  CustomAllocator& operator=(const CustomAllocator&) = delete;
  size_t empty_cache_count_;
  size_t alloc_count_;
  size_t free_count_;
};

void* CustomAlloc(size_t nbytes) {
  return CustomAllocator::GetInstance().Alloc(nbytes);
}

void CustomFree(void* ptr) {
  return CustomAllocator::GetInstance().Free(ptr);
}

TEST(AllocatorTest, CUDAExternalAllocator) {
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.device_id = 0;
  provider_options.do_copy_in_default_stream = true;
  provider_options.alloc = CustomAlloc;
  provider_options.free = CustomFree;
  auto factory = CudaProviderFactoryCreator::Create(&provider_options);
  auto provider = factory->CreateProvider();
  auto allocator = provider->GetAllocator(0, OrtMemTypeDefault);

  size_t size = 8;
  void* cuda_addr_0 = allocator->Alloc(size);
  EXPECT_TRUE(cuda_addr_0);

  // this should trigger an allocation equal to the current total, which should fail initially and gradually fall back
  // to a smaller block.
  size_t next_size = 1024;

  void* cuda_addr_1 = allocator->Alloc(next_size);
  EXPECT_TRUE(cuda_addr_1);
  allocator->Free(cuda_addr_0);
  allocator->Free(cuda_addr_1);

  EXPECT_EQ(CustomAllocator::GetInstance().GetAllocCount(), 2);
  EXPECT_EQ(CustomAllocator::GetInstance().GetFreeCount(), 2);
}
}  // namespace test
}  // namespace onnxruntime
