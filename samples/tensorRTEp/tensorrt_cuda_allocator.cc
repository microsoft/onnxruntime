// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cuda_runtime_api.h>
#include "tensorrt_cuda_allocator.h"

void CUDA_RETURN_IF_ERROR(cudaError_t res);

namespace onnxruntime {
void CUDAAllocator::CheckDevice(bool throw_when_fail) const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call cudaSetDevice instead of the check
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    assert(current_device == CUDAAllocator::GetDeviceId());
  } else if (throw_when_fail) {
    CUDA_RETURN_IF_ERROR(cuda_err);
  }
#else
  ORT_UNUSED_PARAMETER(throw_when_fail);
#endif
}

void CUDAAllocator::SetDevice(bool throw_when_fail) const {
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    int allocator_device_id = CUDAAllocator::GetDeviceId();
    if (current_device != allocator_device_id) {
      cuda_err = cudaSetDevice(allocator_device_id);
    }
  }

  if (cuda_err != cudaSuccess && throw_when_fail) {
    CUDA_RETURN_IF_ERROR(cuda_err);
  }
}

void* CUDAAllocator::Alloc(size_t size) {
  SetDevice(true);
  CheckDevice(true);
  void* p = nullptr;
  if (size > 0) {
    // BFCArena was updated recently to handle the exception and adjust the request size
    CUDA_RETURN_IF_ERROR(cudaMalloc((void**)&p, size));
  }
  return p;
}

void CUDAAllocator::Free(void* p) {
  SetDevice(false);
  CheckDevice(false);  // ignore CUDA failure when free
  cudaFree(p);         // do not throw error since it's OK for cudaFree to fail during shutdown
}

const OrtMemoryInfo* CUDAAllocator::Info() const {
  return mem_info_;
}

void* CUDAPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CUDA_RETURN_IF_ERROR(cudaMallocHost((void**)&p, size));
  }
  return p;
}

void CUDAPinnedAllocator::Free(void* p) {
  CUDA_RETURN_IF_ERROR(cudaFreeHost(p));
}

const OrtMemoryInfo* CUDAPinnedAllocator::Info() const {
  return mem_info_;
}

}  // namespace onnxruntime
