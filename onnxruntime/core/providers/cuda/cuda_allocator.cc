// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_common.h"
#include "cuda_allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "cuda_fence.h"

namespace onnxruntime {

static const CUDAExecutionProvider* GetCUDAExecutionProvider(const SessionState* session_state) {
  return dynamic_cast<const CUDAExecutionProvider*>(
      session_state->GetExecutionProviders().Get(onnxruntime::kCudaExecutionProvider));
}

void CUDAAllocator::CheckDevice() const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call cudaSetDevice instead of the check
  int current_device;
  CUDA_CALL_THROW(cudaGetDevice(&current_device));
  ONNXRUNTIME_ENFORCE(current_device == device_id_);
#endif
}

void* CUDAAllocator::Alloc(size_t size) {
  CheckDevice();
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL_THROW(cudaMalloc((void**)&p, size));
  }
  return p;
}

void CUDAAllocator::Free(void* p) {
  CheckDevice();
  cudaFree(p);  // do not throw error since it's OK for cudaFree to fail during shutdown
}

const ONNXRuntimeAllocatorInfo& CUDAAllocator::Info() const {
  return info_;
}

FencePtr CUDAAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CUDAFence>(GetCUDAExecutionProvider(session_state));
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

const ONNXRuntimeAllocatorInfo& CUDAPinnedAllocator::Info() const {
  static constexpr ONNXRuntimeAllocatorInfo cuda_allocator_info(CUDA_PINNED, ONNXRuntimeDeviceAllocator, 0, ONNXRuntimeMemTypeCPUOutput);
  return cuda_allocator_info;
}

FencePtr CUDAPinnedAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CUDAFence>(GetCUDAExecutionProvider(session_state));
}

}  // namespace onnxruntime
