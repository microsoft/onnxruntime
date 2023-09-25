// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <emscripten.h>

#include "core/framework/session_state.h"
#include "core/providers/js/allocator.h"

namespace onnxruntime {
namespace js {

void* JsCustomAllocator::Alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }

  void* p = EM_ASM_PTR({ return Module.jsepAlloc($0); }, size);
  stats_.num_allocs++;
  stats_.bytes_in_use += size;
  return p;
}

void JsCustomAllocator::Free(void* p) {
  if (p != nullptr) {
    size_t size = (size_t)(void*)EM_ASM_PTR({ return Module.jsepFree($0); }, p);
    stats_.bytes_in_use -= size;
  }
}

void JsCustomAllocator::GetStats(AllocatorStats* stats) {
  *stats = stats_;
}

}  // namespace js
}  // namespace onnxruntime
