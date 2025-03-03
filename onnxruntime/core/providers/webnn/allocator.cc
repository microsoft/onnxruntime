// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webnn/allocator.h"

#include "core/common/safeint.h"

namespace onnxruntime {
namespace webnn {

void* WebNNTensorAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  if (!emscripten::val::module_property("shouldTransferToMLTensor").as<bool>()) {
    // We don't need to transfer the tensor to an MLTensor, so we don't need to allocate an MLTensor id.
    return nullptr;
  }
  void* p = EM_ASM_PTR({ return Module.jsepReserveTensorId(); });
  allocations_[p] = size;
  stats_.num_allocs++;
  stats_.bytes_in_use += SafeInt<int64_t>(size);
  return p;
}

void WebNNTensorAllocator::Free(void* p) {
  if (p == nullptr) {
    return;
  }
  EM_ASM({ Module.jsepReleaseTensorId($0); }, p);
  size_t size = allocations_[p];
  stats_.bytes_in_use -= size;
  allocations_.erase(p);
}

void WebNNTensorAllocator::GetStats(AllocatorStats* stats) {
  *stats = stats_;
}

}  // namespace webnn
}  // namespace onnxruntime
