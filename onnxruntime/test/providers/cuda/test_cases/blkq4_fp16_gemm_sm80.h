/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Module Name:
 *    blkq4_fp16_gemm_sm80.h
 *
 * Abstract:
 *   Bridge between gtest code and gemm kernel implementation.
 *   Gemm kernel requires CUTLASS header files, which causes strange
 *   compilation errors with RE2 header files, which are required
 *   by gtest.
 */

#pragma once

#include <random>

#include "core/util/matrix_layout.h"
#include "core/common/common.h"
#include "core/mickey/blk_q4/f16_prepack_sm80.h"
#include "test/cuda_host/blkq4_fp16_quant_sm80.h"

namespace onnxruntime {
namespace cuda {
namespace test {

Status sm80_supported();

template <
    int block_size,
    bool column_wise_blocking,
    bool small_m,
    bool has_offsets>
void run_blkq4_gemm(int m, int n, int k);

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
