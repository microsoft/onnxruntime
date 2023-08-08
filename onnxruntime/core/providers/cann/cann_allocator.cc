// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cann/cann_call.h"
#include "core/providers/cann/cann_allocator.h"
#include "core/providers/cann/npu_data_transfer.h"

namespace onnxruntime {

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

}  // namespace onnxruntime
