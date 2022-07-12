// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUDA
#include "cuda_tool.h"
// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "nvToolsExt.h"
// Pytorch
#include <ATen/cuda/CUDAContext.h>
// ORT
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/provider_factory_creators.h"

namespace onnxruntime {
namespace lazytensor {

NvtxRange::NvtxRange(const char* name) {
  nvtxRangePush(name);
}
NvtxRange::NvtxRange(const std::string& name) {
  nvtxRangePush(name.c_str());
}
NvtxRange::~NvtxRange() {
  nvtxRangePop();
}

// Wrapper of memory allocation function.
void* CudaAllocDelegate(size_t nbytes) {
  auto allocator = at::cuda::getCUDADeviceAllocator();
  return allocator->raw_allocate(nbytes);
}

// Wrapper of memory de-allocation function.
void CudaFreeDelegate(void* ptr) {
  auto allocator = at::cuda::getCUDADeviceAllocator();
  allocator->raw_deallocate(ptr);
}

void CUDAExecutionProviderPool::Initialize() {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  for (int i = 0; i < device_count; ++i) {
    OrtCUDAProviderOptionsV2 provider_options{};
    provider_options.device_id = i;
    provider_options.do_copy_in_default_stream = true;
    provider_options.alloc = reinterpret_cast<void*>(&CudaAllocDelegate);
    provider_options.free = reinterpret_cast<void*>(CudaFreeDelegate);
    auto factory = onnxruntime::CudaProviderFactoryCreator::Create(&provider_options);
    cuda_execution_providers_.emplace_back(std::move(factory->CreateProvider()));
  }
}

}  // namespace lazytensor
}  // namespace onnxruntime
#endif
