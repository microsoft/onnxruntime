// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "rocm_allocator.h"
#include "rocm_common.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "rocm_fence.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

static const GPUDataTransfer* GetGPUDataTransfer(const SessionState* session_state) {
  OrtDevice gpu_device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0);
  OrtDevice cpu_device;
  return static_cast<const GPUDataTransfer*>(session_state->GetDataTransferMgr().GetDataTransfer(gpu_device, cpu_device));
}

void ROCMAllocator::CheckDevice(bool throw_when_fail) const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call hipSetDevice instead of the check
  int current_device;
  auto hip_err = hipGetDevice(&current_device);
  if (hip_err == hipSuccess) {
    ORT_ENFORCE(current_device == Info().id);
  } else if (throw_when_fail) {
    HIP_CALL_THROW(hip_err);
  }
#else
  ORT_UNUSED_PARAMETER(throw_when_fail);
#endif
}

void* ROCMAllocator::Alloc(size_t size) {
  CheckDevice(true);
  void* p = nullptr;
  if (size > 0) {
    //BFCArena was updated recently to handle the exception and adjust the request size
    HIP_CALL_THROW(hipMalloc((void**)&p, size));
  }
  return p;
}

void ROCMAllocator::Free(void* p) {
  CheckDevice(false);  // ignore HIP failure when free
  hipFree(p);         // do not throw error since it's OK for hipFree to fail during shutdown
}

FencePtr ROCMAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<ROCMFence>(GetGPUDataTransfer(session_state));
}

void* ROCMExternalAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = alloc_(size);

    // review(codemzs): ORT_ENFORCE does not seem appropiate.
    ORT_ENFORCE(p != nullptr);

  }

  return p;
}

void ROCMExternalAllocator::Free(void* p) {
  free_(p);
}

void* ROCMPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    HIP_CALL_THROW(hipHostMalloc((void**)&p, size));
  }
  return p;
}

void ROCMPinnedAllocator::Free(void* p) {
  HIP_CALL_THROW(hipHostFree(p));
}

FencePtr ROCMPinnedAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<ROCMFence>(GetGPUDataTransfer(session_state));
}

}  // namespace onnxruntime
