// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUDA
#include "cuda_tool.h"
// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#include "nvtx3/nvToolsExt.h"
#else
#include "nvToolsExt.h"
#endif
// Pytorch
#include <ATen/cuda/CUDAContext.h>
// ORT
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/provider_factory_creators.h"
#include "orttraining/python/orttraining_pybind_common.h"

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
    onnxruntime::ProviderOptions options;
    options["device_id"] = std::to_string(i);
    options["do_copy_in_default_stream"] = "true";
    options["gpu_external_alloc"] = std::to_string(reinterpret_cast<size_t>(&CudaAllocDelegate));
    options["gpu_external_free"] = std::to_string(reinterpret_cast<size_t>(&CudaFreeDelegate));

    ProviderInfo_CUDA* cuda_provider_info = TryGetProviderInfo_CUDA();
    CUDAExecutionProviderInfo info;
    cuda_provider_info->CUDAExecutionProviderInfo__FromProviderOptions(options, info);
    cuda_execution_providers_.emplace_back(std::move(cuda_provider_info->CreateExecutionProviderFactory(info)->CreateProvider()));
  }
}

}  // namespace lazytensor
}  // namespace onnxruntime
#endif
