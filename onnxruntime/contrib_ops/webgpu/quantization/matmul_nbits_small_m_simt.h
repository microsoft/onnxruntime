// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "core/providers/webgpu/compute_context.h"

namespace onnxruntime::contrib::webgpu {

struct SmallMSimtConfig {
  uint32_t bn = 16;
  uint32_t bk = 512;
  uint32_t a_block_padding = 0;
};

Status ApplySmallMSimtMatMulNBits(const Tensor* a, const Tensor* b, const Tensor* scales,
                                  const Tensor* zero_points, const Tensor* bias, uint32_t M, uint32_t N,
                                  uint32_t K, uint32_t block_size, uint32_t batch_count, uint32_t weight_index,
                                  const Tensor* weight_index_indirect, uint32_t dispatch_M, uint32_t nbits,
                                  onnxruntime::webgpu::ComputeContext& context, Tensor* y,
                                  const SmallMSimtConfig& config);

}  // namespace onnxruntime::contrib::webgpu
