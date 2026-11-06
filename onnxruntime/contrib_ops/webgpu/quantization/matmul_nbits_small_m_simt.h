// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "core/providers/webgpu/compute_context.h"

namespace onnxruntime::contrib::webgpu {

bool CanApplySmallMSimtMatMulNBits(const Tensor* a, uint32_t N, uint32_t K,
                                   uint32_t block_size, int64_t nbits,
                                   bool has_weight_idx_indirect, uint32_t override_M,
                                   const onnxruntime::webgpu::ComputeContext& context);

Status ApplySmallMSimtMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                  const Tensor* zero_points, const Tensor* bias,
                                  uint32_t N, uint32_t K, uint32_t weight_index,
                                  onnxruntime::webgpu::ComputeContext& context, Tensor* y);

}  // namespace onnxruntime::contrib::webgpu
