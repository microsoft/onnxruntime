// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "migraphx_inc.h"
#include "hip_allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "hip_fence.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

static const GPUDataTransfer* GetGPUDataTransfer(const SessionState* session_state) {
  OrtDevice gpu_device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0);
  OrtDevice cpu_device;
  return dynamic_cast<const GPUDataTransfer*>(session_state->GetDataTransferMgr().GetDataTransfer(gpu_device, cpu_device));
}

void HIPAllocator::CheckDevice() const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call hipSetDevice instead of the check
  int current_device;
  hipGetDevice(&current_device);
  ORT_ENFORCE(current_device == info_.id);
#endif
}

void* HIPAllocator::Alloc(size_t size) {
  CheckDevice();
  void* p = nullptr;
  if (size > 0) {
    hipMalloc((void**)&p, size);
  }
  return p;
}

void HIPAllocator::Free(void* p) {
  CheckDevice();
  hipFree(p);  // do not throw error since it's OK for hipFree to fail during shutdown
}

FencePtr HIPAllocator::CreateFence(const SessionState* session_state) {
 return std::make_shared<HIPFence>(GetGPUDataTransfer(session_state));
}

void* HIPPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    hipHostMalloc((void**)&p, size);
  }
  return p;
}

void HIPPinnedAllocator::Free(void* p) {
  hipHostFree(p);
}

FencePtr HIPPinnedAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<HIPFence>(GetGPUDataTransfer(session_state));
}

}  // namespace onnxruntime
