// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <tvm/runtime/device_api.h>

#include "tvm_allocator.h"
#include "core/framework/session_state.h"
#include "xpu_data_transfer.h"

namespace onnxruntime {
namespace tvm {

void* TVMAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    DLDataType dl_type{kDLInt, 8, 1};
    int err = TVMDeviceAllocDataSpace(ctx, size, ::tvm::runtime::kAllocAlignment, dl_type, reinterpret_cast<void**>(&p));
    CHECK_EQ(err, 0);
    return p;
  }
  return p;
}

void TVMAllocator::Free(void* p) {
  TVMDeviceFreeDataSpace(ctx, p);
}

}  // namespace tvm
}  // namespace onnxruntime
