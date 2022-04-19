// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tvm_allocator.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "xpu_data_transfer.h"


namespace onnxruntime {
namespace tvm {

void* TVMAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    DLDataType dl_type{kDLInt, 8, 1};
    int err = TVMDeviceAllocDataSpace(ctx, size, 128, dl_type, (void**)&p);
    CHECK_EQ(err, 0);
    return p;
  }
  return p;
}

void TVMAllocator::Free(void* p) {
    TVMDeviceFreeDataSpace(ctx, p);
}

}   // namespace tvm
}   // namespace onnxruntime
