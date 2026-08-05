// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// ROCm GatherBlockQuantized kernel — GPU implementation.
// Arch-portable HIP, no CUTLASS / CK / MFMA.  Runs on gfx900 and newer.

#pragma once

#include <hip/hip_runtime.h>
#include "core/framework/int4.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

struct GatherBlockQuantizedParam {
  hipStream_t stream;
  int64_t after_gather_dim;
  int64_t gather_axis_dim;
  int64_t ind_dim;
  int64_t bits;
  int64_t block_size;
  int64_t gather_axis;
  int64_t N;
};

// Typed launch wrapper — one explicit instantiation per (T1, T2, Tind) combo.
template <typename T1, typename T2, typename Tind>
void LaunchGatherBlockQuantizedKernel(
    const T1* data,
    const Tind* indices,
    const T2* scales,
    const T1* zero_points,  // may be nullptr
    T2* output,
    GatherBlockQuantizedParam param);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
