// Copyright (c) Microsoft Corporation. All rights reserved.

#include "core/providers/cuda/plugin_ep/allocator.h"
#include "core/providers/cuda/plugin_ep/utils.h"

namespace cuda_plugin_ep {

namespace {
void SetDevice(const OrtMemoryDevice& device) {
  int current_device;
  auto device_id = Shared::ep_api->MemoryDevice_GetDeviceId(&device);

  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    int allocator_device_id = device_id;
    if (current_device != allocator_device_id) {
      cuda_err = cudaSetDevice(allocator_device_id);
    }
  }

  CUDA_THROW_IF_ERROR(cuda_err);
}

}  // namespace

CudaOrtAllocator::CudaOrtAllocator(const OrtMemoryInfo* mem_info, const OrtApi& api)
    : memory_info_{*mem_info},
      memory_device_{*Shared::ep_api->MemoryInfo_GetMemoryDevice(mem_info)} {
  version = ORT_API_VERSION;
  Info = InfoImpl;
  Reserve = AllocImpl;  // no special behavior for Reserve so use AllocImpl
  GetStats = nullptr;   // GetStatsImpl. The CUDA allocators don't have stats currently so we can skip.

  const OrtMemoryDevice* mem_device = Shared::ep_api->MemoryInfo_GetMemoryDevice(mem_info);

  if (Shared::ep_api->MemoryDevice_GetMemoryType(mem_device) == OrtDeviceMemoryType_HOST_ACCESSIBLE) {
    Alloc = PinnedAllocImpl;
    Free = PinnedFreeImpl;
  } else {
    Alloc = AllocImpl;
    Free = FreeImpl;
  }
}

void* CudaOrtAllocator::AllocImpl(struct OrtAllocator* this_, size_t size) {
  auto& impl = *static_cast<CudaOrtAllocator*>(this_);

  // TODO: This should not be necessary unless the allocator is being used in different sessions.
  //       Should we have a flag for the shared allocator so we can skip the cudaSetDevice call
  //       if it's a per-session allocator?
  SetDevice(impl.memory_device_);

  void* p = nullptr;
  if (size > 0) {
    CUDA_THROW_IF_ERROR(cudaMalloc((void**)&p, size));
  }
  return p;
}

void CudaOrtAllocator::FreeImpl(struct OrtAllocator* this_, void* p) {
  auto& impl = *static_cast<CudaOrtAllocator*>(this_);
  SetDevice(impl.memory_device_);

  // do not throw error since it's OK for cudaFree to fail during shutdown
  cudaFree(p);
}

const struct OrtMemoryInfo* CudaOrtAllocator::InfoImpl(const struct OrtAllocator* this_) {
  const CudaOrtAllocator& impl = *static_cast<const CudaOrtAllocator*>(this_);
  return &impl.memory_info_;
}

void* CudaOrtAllocator::PinnedAllocImpl(struct OrtAllocator* this_, size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CUDA_THROW_IF_ERROR(cudaMallocHost((void**)&p, size));
  }
  return p;
}

void CudaOrtAllocator::PinnedFreeImpl(struct OrtAllocator* this_, void* p) {
  CUDA_THROW_IF_ERROR(cudaFreeHost(p));
}

}  // namespace cuda_plugin_ep
