// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/bert/paged/config.h"

namespace onnxruntime::contrib::paged {

namespace constant {
inline constexpr const int WarpSize = 32;
}  // namespace constant

__forceinline__ __device__ void
schedule_barrier() {
  // do nothing
}

__noinline__ __device__ static void
enforce_uniform() {
  asm("ret.uni;\n" ::);
  // __syncwarp();  // be old-school
}

}  // namespace onnxruntime::contrib::paged
