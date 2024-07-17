// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/rocm/bert/paged/config.h"
#include "hip/hip_fp16.h"

// https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.def

namespace onnxruntime::contrib::paged {

namespace constant {
inline constexpr const int WarpSize = 64;
}  // namespace constant

__forceinline__ __device__ void
schedule_barrier() {
  __builtin_amdgcn_sched_barrier(0);
}

__forceinline__ __device__ void
enforce_uniform() {
  // do nothing
}

__forceinline__ __device__ float
hfma2(const half2& a, const half2& b, float& c) {
#if __HIP_DEVICE_COMPILE__
#if PAGED_INNER_PRODUCT_FP16_ARITHMETIC_FP32_ACC
  // asm volatile("\n v_dot2c_f32_f16 %0, %1, %2\n s_nop 4" : "+v"(c) : "v"(a), "v"(b));
  c = __builtin_amdgcn_fdot2(a, b, c, false);
  return c;
#else
  // static_assert(false, "not implemented");
#endif
  return std::numeric_limits<float>::quiet_NaN();
#endif
}

}  // namespace onnxruntime::contrib::paged
