// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_allocator.h"
#include "cuda_common.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "cuda_fence.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

static const GPUDataTransfer* GetGPUDataTransfer(const SessionState* session_state) {
  OrtDevice gpu_device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0);
  OrtDevice cpu_device;
  return static_cast<const GPUDataTransfer*>(session_state->GetDataTransferMgr().GetDataTransfer(gpu_device, cpu_device));
}

void CUDAAllocator::CheckDevice(bool throw_when_fail) const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call cudaSetDevice instead of the check
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    ORT_ENFORCE(current_device == Info().id);
  } else if (throw_when_fail) {
    CUDA_CALL_THROW(cuda_err);
  }
#else
  ORT_UNUSED_PARAMETER(throw_when_fail);
#endif
}

void CUDAAllocator::SetDevice(bool throw_when_fail) const {
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    int allocator_device_id = Info().id;
    if (current_device != allocator_device_id) {
      cuda_err = cudaSetDevice(allocator_device_id);
    }
  }

  if (cuda_err != cudaSuccess && throw_when_fail) {
    CUDA_CALL_THROW(cuda_err);
  }
}

void* CUDAAllocator::Alloc(size_t size) {
  SetDevice(true);
  CheckDevice(true);
  void* p = nullptr;
  if (size > 0) {
    //BFCArena was updated recently to handle the exception and adjust the request size
    CUDA_CALL_THROW(cudaMalloc((void**)&p, size));
  }
  return p;
}

void CUDAAllocator::Free(void* p) {
  SetDevice(false);
  CheckDevice(false);  // ignore CUDA failure when free
  cudaFree(p);         // do not throw error since it's OK for cudaFree to fail during shutdown
}

void* CUDAExternalAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = alloc_(size);

    // review(codemzs): ORT_ENFORCE does not seem appropiate.
    ORT_ENFORCE(p != nullptr);

  }

  return p;
}

void CUDAExternalAllocator::Free(void* p) {
  free_(p);
}

FencePtr CUDAAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CUDAFence>(GetGPUDataTransfer(session_state));
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

FencePtr CUDAPinnedAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CUDAFence>(GetGPUDataTransfer(session_state));
}

}  // namespace onnxruntime
