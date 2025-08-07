// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {

Status GemmMatMul(
    cudaStream_t stream, bool has_bias, bool has_scales,
    int32_t dtype_A, int32_t dtype_B,
    int32_t dtype_C, int32_t dtype_Y,
    bool trans_A, bool trans_B, const void* p_input_a, const void* p_input_b,
    const void* p_input_c, const void* p_scale_a, const void* p_scale_b,
    const void* p_scale_y, void* p_output_y, int M, int N, int K, int lda,
    int ldb, int ldd, bool row_major_compute, int64_t sm_count, int /* cublasLtEpilogue_t */ epilogue,
    float alpha, float beta);

}  // namespace cuda
}  // namespace onnxruntime
