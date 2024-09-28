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

__host__ __device__ inline constexpr int
compute_min_resident_blocks(int threads_per_block, int registers_per_thread) {
  // at warp level register allocation granularity, stable date back to sm_52
  constexpr int register_alloc_unit = 256;

  // hardware resource of registers, stable date back to sm_60
  constexpr int max_registers_per_block = 65536;

  int registers_per_warp = ((registers_per_thread * constant::WarpSize - 1) / register_alloc_unit + 1) * register_alloc_unit;
  int resident_warps = max_registers_per_block / registers_per_warp;
  int warps_per_block = threads_per_block / constant::WarpSize;
  return resident_warps / warps_per_block;
}

}  // namespace onnxruntime::contrib::paged


// original second parameter of __launch_bounds__ constraints the desired min_resident_blocks
#define PAGED_LAUNCH_BOUNDS(threads_per_block, registers_per_thread) \
  __launch_bounds__(threads_per_block, compute_min_resident_blocks(threads_per_block, registers_per_thread))
