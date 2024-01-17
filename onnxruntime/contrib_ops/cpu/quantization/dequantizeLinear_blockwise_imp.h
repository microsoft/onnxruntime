// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define DequantizeLinearBlockWise operator, it is basically
// dequantize input tensor and unpack it into float/half tensor.
//

#pragma once

#include "core/mlas/inc/mlas_q4.h"
namespace onnxruntime {
namespace contrib {
namespace GPTQPacking {

template <typename SCALE_TYPE>
void DequantWeightNbitGidx(concurrency::ThreadPool* pool,
                           const int32_t* qweight_i32_i, const SCALE_TYPE* scale_fp16,
                           const uint32_t* qzeros_i32_i, const int32_t* g_dix,
                           SCALE_TYPE* b_fp16,
                           uint32_t mat_k, uint32_t mat_n, int bits);

template <typename SCALE_TYPE, typename ZEROT>
void DequantWeightNbit(
    concurrency::ThreadPool* pool,
    const int32_t* qweight_i32,
    const SCALE_TYPE* scale,
    const ZEROT* qzeros_i32,
    SCALE_TYPE* output,
    uint32_t mat_k,
    uint32_t mat_n,
    uint32_t bits,
    uint32_t groupsize);
template <typename ZERO_TYPE>
void GeneralDequant(concurrency::ThreadPool* pool, const int32_t* qweight_i32, const float* scale,
                    const ZERO_TYPE* qzeros_i32, const int32_t* g_idx,
                    float* output, uint32_t mat_k, uint32_t mat_n, int bits, int group_size);

}  // namespace GPTQPacking
}  // namespace contrib
}  // namespace onnxruntime
