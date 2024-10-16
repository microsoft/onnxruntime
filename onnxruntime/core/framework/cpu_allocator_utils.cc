// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/cpu_allocator_utils.h"

#include <absl/base/config.h>

namespace onnxruntime {

bool ShouldCpuAllocatorUseArena([[maybe_unused]] bool is_arena_requested) {
#if defined(USE_JEMALLOC) || defined(USE_MIMALLOC)
  // We use these allocators instead of the arena.
  return false;
#elif defined(ABSL_HAVE_ADDRESS_SANITIZER)
  // Using the arena may hide memory issues. Disable it in an ASan build.
  return false;
#else
  // Disable the arena for 32-bit builds because it may run into an infinite loop on integer overflow.
  if constexpr (sizeof(void*) == 4) {
    return false;
  } else {
    return is_arena_requested;
  }
#endif
}

}  // namespace onnxruntime
