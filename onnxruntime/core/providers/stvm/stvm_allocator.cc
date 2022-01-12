// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "stvm_allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "xpu_data_transfer.h"


namespace onnxruntime {


void* STVMAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    DLDataType dl_type{kDLInt, 8, 1};
    int err = TVMDeviceAllocDataSpace(ctx, size, 128, dl_type, (void**)&p);
    CHECK_EQ(err, 0);
    return p;
  }
  return p;
}

void STVMAllocator::Free(void* p) {
    TVMDeviceFreeDataSpace(ctx, p);
}

}  // namespace onnxruntime
