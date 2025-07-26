// Copyright (c) Microsoft Corporation. All rights reserved.

#include "core/providers/cuda/plugin_ep/allocator.h"
#include "core/providers/cuda/plugin_ep/utils.h"

namespace cuda_plugin_ep {

namespace {
OrtStatus* CheckDevice(const OrtMemoryDevice& device) {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call cudaSetDevice instead of the check
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  EP_ENFORCE(cuda_err == cudaSuccess, "Failed to get current CUDA device: " + std::to_string(cuda_err));
  EP_ENFORCE(current_device == Shared::ep_api->MemoryDevice_GetDeviceId(&device), __FUNCTION__);
#endif
}

OrtStatus* SetDevice(const OrtMemoryDevice& device) {
  int current_device;
  auto device_id = Shared::ep_api->MemoryDevice_GetDeviceId(&device);

  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    int allocator_device_id = device_id;
    if (current_device != allocator_device_id) {
      cuda_err = cudaSetDevice(allocator_device_id);
    }
  }

  CUDA_RETURN_IF_ERROR(cuda_err);

  return nullptr;
}


}  // namespace

/*
void* CUDAAllocator::Alloc(size_t size) {
  SetDevice(true);
  CheckDevice(true);
  void* p = nullptr;
  if (size > 0) {
    // BFCArena was updated recently to handle the exception and adjust the request size
    CUDA_CALL_THROW(cudaMalloc((void**)&p, size));
  }
  return p;
}

void CUDAAllocator::Free(void* p) {
  SetDevice(false);
  CheckDevice(false);  // ignore CUDA failure when free
  cudaFree(p);         // do not throw error since it's OK for cudaFree to fail during shutdown
}

void* CUDAPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL_THROW(cudaMallocHost((void**)&p, size));
  }
  return p;
}

void CUDAPinnedAllocator::Free(void* p) {
  CUDA_CALL_THROW(cudaFreeHost(p));
}
*/

CudaOrtAllocator::CudaOrtAllocator(const OrtMemoryInfo* mem_info, const OrtApi& api)
    : memory_info_{*mem_info} {
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

void* ORT_API_CALL CudaOrtAllocator::AllocImpl(struct OrtAllocator* this_, size_t size) {
  auto& impl = *static_cast<CudaOrtAllocator*>(this_);
  SetDevice();
  CheckDevice(true);
  void* p = nullptr;
  if (size > 0) {
    // BFCArena was updated recently to handle the exception and adjust the request size
    THROW_IF_ERROR(cudaMalloc((void**)&p, size));
  }
  return p;
}

void ORT_API_CALL CudaOrtAllocator::FreeImpl(struct OrtAllocator* this_, void* p) {
  auto& impl = *static_cast<CudaOrtAllocator*>(this_);
  impl.allocator_->Free(p);
}

const struct OrtMemoryInfo* ORT_API_CALL CudaOrtAllocator::InfoImpl(const struct OrtAllocator* this_) {
  const CudaOrtAllocator& impl = *static_cast<const CudaOrtAllocator*>(this_);
  return impl.memory_info_;
}

void* CUDAPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL_THROW(cudaMallocHost((void**)&p, size));
  }
  return p;
}

void CUDAPinnedAllocator::Free(void* p) {
  CUDA_CALL_THROW(cudaFreeHost(p));
}

}  // namespace cuda_plugin_ep
