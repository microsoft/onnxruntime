/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qlutgemm.h

Abstract:

    This module includes kernel function prototypes and helper functions for
    implementing LUT-based GEMM.
--*/

#pragma once

#include "mlas_qnbit.h"
#include "mlasi.h"

/**
 * @brief Parameters for TMAC kernel
 */
struct MlasTMACKernelParams {
    size_t g;
    size_t ngroups_per_elem;
    size_t q_group_size;
    size_t act_group_size;

    size_t kfactor;
    size_t bits;
    size_t actk;
    size_t bm;
    size_t simd_n_in;
    size_t simd_n_out;
    size_t chunk_n;
    size_t n_tiles_num;

    bool has_scale;
    bool has_zero_point;
    bool one_scale;
};

/**
 * Retrieves the T-MAC kernel configuration for a given GEMM problem.
 * Returns the parameters by value to ensure thread-safety across concurrent calls.
 */
MlasTMACKernelParams
MlasGetLutGemmKernelParams(size_t M, size_t N, size_t nbits, size_t block_size, bool has_zero_point);

typedef void(MLAS_QNBIT_GEMM_LUT_GEN)(
    const float* b,
    int8_t* qlut,
    float* lut_scales,
    float* lut_biases,
    size_t M,
    size_t K,
    size_t N,
    size_t act_group_size,
    size_t lut_stride        // Stride (in bytes) between consecutive LUT entries along the batch dimension.
);

typedef void(MLAS_QNBIT_LUT_GEMM_COMPUTE)(
    const uint8_t* A,
    const float* Scales,
    const int8_t* LUT,
    const float* LUT_Scales,
    const float* LUT_Biases,
    float* C,
    int K,
    int M,                  // Batch size (current activation rows).
    int N,                  // Number of output features to compute in this tile/chunk.
    int TotalN,             // Total number of output features in the weights (used for parameter mapping).
    size_t BlkLen,
    bool HasZeroPoint
);

//
// Kernel dispatch structure.
//
// NOTE: This name must match the forward declaration in mlasi.h:
//   struct MLAS_QNBIT_LUT_GEMM_DISPATCH;
// Keep it minimal for now; extend with function pointers as kernels are added.
struct MLAS_QNBIT_LUT_GEMM_DISPATCH {
    // Intentionally empty placeholder; add members as needed.
    MLAS_QNBIT_GEMM_LUT_GEN* GenerateLUT = nullptr;

    MLAS_QNBIT_LUT_GEMM_COMPUTE* ComputeGemm = nullptr;
};
