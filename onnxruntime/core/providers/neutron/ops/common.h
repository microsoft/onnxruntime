// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace neutron {

#define ALIGN16_SIZE(size) ((size + 0xf) & (~0xf))

void PrepareForQDQ(const TensorShape& input_shape,
                   const Tensor& scale,
                   const Tensor* zero_point_ptr,
                   int64_t axis,
                   int64_t& block_count,
                   int64_t& broadcast_dim,
                   int64_t& block_size);

uint32_t ScaleToNeutron(float scale_data);

std::tuple<int, int, int>
TilingSolver(int embeddings_in,
             int groupSize,
             int resNumBytes,
             int weightBits,
             bool decodeWeights = true,
             bool useDecodeBias = true,
             int MACS = 16,
             int neutrons = 4,
             int tcm_size = 1024 * 1024,
             int tcm_banks = 16);

void OrganizeWeightsData(const int8_t* weights,
                         int8_t* output,
                         int rowsB,
                         int colsB,
                         int channelDensity,
                         int numNeutrons,
                         int weightBits = 4,
                         int MACs = 16,
                         bool isTransposed = false);

int32_t
GetMatmulTypeFlag(bool packed, bool signedData);
}  // namespace neutron
}  // namespace onnxruntime
