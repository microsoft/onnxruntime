// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/compute_context.h"

namespace onnxruntime::contrib::webgpu {

enum class MatMulNBitsTile {
  None,
  Register,
  Matrix,
};

// Requires an Apple adapter with enabled, fixed-width 32-lane subgroups.
// matrix_available must reflect an enabled FP32 8x8x8 subgroup matrix configuration.
// Restrict the initial dispatch range to measured, fully occupied tiles.
constexpr MatMulNBitsTile SelectAppleMatMulNBitsTile(uint32_t m, uint32_t n, uint32_t k,
                                                     bool matrix_available, bool fp32_accumulation) {
  if (m % 64 != 0 || n < 4096 || k < 4096 || n % 64 != 0 || k % 32 != 0) {
    return MatMulNBitsTile::None;
  }
  if (matrix_available && m >= 128 && m <= 512) {
    return MatMulNBitsTile::Matrix;
  }
  return m >= 256 && !fp32_accumulation ? MatMulNBitsTile::Register : MatMulNBitsTile::None;
}

Status TryApplyMatMulNBitsTile(const Tensor* a, const Tensor* b, const Tensor* scales, Tensor* y,
                               uint32_t m, uint32_t n, uint32_t k,
                               onnxruntime::webgpu::ComputeContext& context, bool& handled);

}  // namespace onnxruntime::contrib::webgpu
