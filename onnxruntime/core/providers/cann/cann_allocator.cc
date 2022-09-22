// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/cann_call.h"
#include "core/providers/cann/cann_allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/providers/cann/cann_fence.h"
#include "core/providers/cann/npu_data_transfer.h"

namespace onnxruntime {

static const NPUDataTransfer* GetNPUDataTransfer(const SessionState* session_state) {
  OrtDevice npu_device(OrtDevice::NPU, OrtDevice::MemType::DEFAULT, 0);
  OrtDevice cpu_device;
  return static_cast<const NPUDataTransfer*>(
      session_state->GetDataTransferMgr().GetDataTransfer(npu_device, cpu_device));
}

void* CANNAllocator::Alloc(size_t size) {
  void* p = nullptr;
  aclrtMemMallocPolicy policy = ACL_MEM_MALLOC_HUGE_FIRST;
  if (size > 0) {
    CANN_CALL_THROW(aclrtMalloc(reinterpret_cast<void**>(&p), size, policy));
  }
  return p;
}

void CANNAllocator::Free(void* p) {
  aclrtFree(p);
}

FencePtr CANNAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CANNFence>(GetNPUDataTransfer(session_state));
}

void* CANNPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CANN_CALL_THROW(aclrtMallocHost(reinterpret_cast<void**>(&p), size));
  }
  return p;
}

void CANNPinnedAllocator::Free(void* p) {
  CANN_CALL_THROW(aclrtFreeHost(p));
}

FencePtr CANNPinnedAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CANNFence>(GetNPUDataTransfer(session_state));
}

}  // namespace onnxruntime
